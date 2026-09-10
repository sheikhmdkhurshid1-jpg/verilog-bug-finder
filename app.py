"""
Verilog Bug Finder - Free MVP
------------------------------
A 100% free, offline Streamlit tool for chip design engineers.
No paid API key required - all analysis runs locally using
rule-based static analysis (regex + pattern matching) tuned to
catch the most common Verilog bugs, timing/simulation risks,
and style issues.

Run locally:
    pip install streamlit
    streamlit run app.py
"""

import re
import streamlit as st
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------
@dataclass
class Issue:
    line_no: int
    line_text: str
    severity: str      # "Bug", "Timing Risk", "Style"
    message: str
    suggestion: str


# ---------------------------------------------------------------------
# Helper: strip comments so they don't create false positives
# ---------------------------------------------------------------------
def strip_comments(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code


# ---------------------------------------------------------------------
# Core analyzer - each function returns a list of Issue objects
# ---------------------------------------------------------------------
def find_blocking_in_sequential(lines: List[str]) -> List[Issue]:
    """Blocking (=) assignments inside an always @(posedge/negedge) block
    are a classic source of simulation-vs-synthesis mismatches."""
    issues = []
    in_seq_block = False
    depth = 0
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if re.search(r"always\s*@\s*\(\s*(posedge|negedge)", line):
            in_seq_block = True
            depth = 0
        if in_seq_block:
            depth += line.count("begin") - line.count("end")
            if re.search(r"(?<![<>=!])=(?!=)", line) and "<=" not in line:
                # exclude declarations like reg [7:0] a = 0; (rare in body, but check)
                if not re.match(r"^\s*(reg|wire|integer|logic)\b", line):
                    issues.append(Issue(
                        i, raw.strip(), "Bug",
                        "Blocking assignment ('=') used inside a clocked (sequential) always block.",
                        "Use non-blocking assignment ('<=') for all signals driven inside "
                        "posedge/negedge blocks to avoid race conditions and sim/synth mismatch."
                    ))
            if depth <= 0 and "end" in line:
                in_seq_block = False
    return issues


def find_nonblocking_in_combinational(lines: List[str]) -> List[Issue]:
    """Non-blocking (<=) assignments inside always @(*) combinational blocks
    can cause simulation mismatches."""
    issues = []
    in_comb_block = False
    depth = 0
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if re.search(r"always\s*@\s*\(\s*\*\s*\)", line) or re.search(r"always_comb", line):
            in_comb_block = True
            depth = 0
        if in_comb_block:
            depth += line.count("begin") - line.count("end")
            if "<=" in line:
                issues.append(Issue(
                    i, raw.strip(), "Bug",
                    "Non-blocking assignment ('<=') used inside a combinational always block.",
                    "Use blocking assignment ('=') in combinational logic to model it correctly."
                ))
            if depth <= 0 and "end" in line:
                in_comb_block = False
    return issues


def find_incomplete_case_if(lines: List[str]) -> List[Issue]:
    """Detects case statements without a default, and if without else,
    both common causes of unintended latch inference."""
    issues = []
    joined = "\n".join(lines)
    for m in re.finditer(r"case\s*\(.*?\)(.*?)endcase", joined, flags=re.DOTALL):
        block = m.group(1)
        if "default" not in block:
            line_no = joined[:m.start()].count("\n") + 1
            issues.append(Issue(
                line_no, "case ( ... ) ... endcase", "Bug",
                "case statement has no 'default' branch.",
                "Add a 'default:' case to fully specify outputs and prevent unintended latch inference."
            ))
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if re.match(r"^\s*if\s*\(.*\)\s*$", line):
            # look ahead a few lines for a matching else at same rough position
            snippet = "\n".join(lines[i:i + 6])
            if "else" not in snippet:
                issues.append(Issue(
                    i, raw.strip(), "Bug",
                    "if-statement without a following else branch.",
                    "Add an 'else' branch (or ensure all outputs have a default value before the if) "
                    "to avoid unintended latch inference in combinational logic."
                ))
    return issues


def find_sensitivity_list_issues(lines: List[str]) -> List[Issue]:
    """Old-style always @(a, b) blocks that are missing a signal used inside."""
    issues = []
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        m = re.match(r"always\s*@\s*\(([^*]+?)\)", line)
        if m and "posedge" not in line and "negedge" not in line and "*" not in m.group(1):
            issues.append(Issue(
                i, raw.strip(), "Timing Risk",
                "Explicit sensitivity list used for combinational logic.",
                "Use always @(*) (or always_comb in SystemVerilog) so the tool infers the full "
                "sensitivity list automatically and avoids simulation/synthesis mismatches from "
                "a missing signal."
            ))
    return issues


def find_unsized_constants(lines: List[str]) -> List[Issue]:
    """Unsized numeric literals (e.g. assign a = 5;) can create unexpected bit widths."""
    issues = []
    pattern = re.compile(r"=\s*\d+\s*;")
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if pattern.search(line) and "'" not in line:
            issues.append(Issue(
                i, raw.strip(), "Style",
                "Unsized decimal constant used in an assignment.",
                "Specify an explicit width and base, e.g. 8'd5 instead of 5, to avoid unintended "
                "bit-width truncation or extension."
            ))
    return issues


def find_multiple_drivers(lines: List[str]) -> List[Issue]:
    """Very rough heuristic: same signal driven by more than one always block
    or by both an always block and a continuous assign."""
    issues = []
    joined = "\n".join(lines)
    assign_targets = re.findall(r"assign\s+([a-zA-Z_]\w*)", joined)
    always_targets = re.findall(r"^\s*([a-zA-Z_]\w*)\s*(?:<=|=)", joined, flags=re.MULTILINE)
    overlap = set(assign_targets) & set(always_targets)
    for sig in overlap:
        line_no = joined.find(f"assign {sig}")
        line_no = joined[:line_no].count("\n") + 1 if line_no != -1 else 1
        issues.append(Issue(
            line_no, f"assign {sig} = ...;", "Bug",
            f"Signal '{sig}' appears to be driven both by a continuous 'assign' and inside an "
            f"always block.",
            f"Drive '{sig}' from only one source (either 'assign' or an always block) to avoid "
            f"a multiple-driver conflict."
        ))
    return issues


def find_missing_reset(lines: List[str]) -> List[Issue]:
    """Sequential always blocks with no reset signal in the sensitivity list."""
    issues = []
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        m = re.search(r"always\s*@\s*\(\s*(posedge|negedge)\s+\w+\s*\)", line)
        if m and "reset" not in line.lower() and "rst" not in line.lower():
            issues.append(Issue(
                i, raw.strip(), "Timing Risk",
                "Clocked always block has no reset (rst/reset) in its sensitivity list.",
                "Add an asynchronous or synchronous reset (e.g. 'or posedge rst') so registers "
                "start in a known state at power-up and after a system reset."
            ))
    return issues


def find_long_combinational_chain(lines: List[str]) -> List[Issue]:
    """Flags very long single assign expressions as a potential timing/critical-path risk."""
    issues = []
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if line.startswith("assign") and len(line) > 120:
            issues.append(Issue(
                i, raw.strip()[:80] + "...", "Timing Risk",
                "Very long single combinational expression.",
                "Consider pipelining or breaking this into intermediate signals to shorten the "
                "combinational critical path and ease timing closure."
            ))
    return issues


ANALYZERS = [
    find_blocking_in_sequential,
    find_nonblocking_in_combinational,
    find_incomplete_case_if,
    find_sensitivity_list_issues,
    find_unsized_constants,
    find_multiple_drivers,
    find_missing_reset,
    find_long_combinational_chain,
]


def analyze_verilog(code: str) -> List[Issue]:
    clean = strip_comments(code)
    lines = clean.split("\n")
    all_issues: List[Issue] = []
    for fn in ANALYZERS:
        try:
            all_issues.extend(fn(lines))
        except Exception:
            # never let one rule crash the whole app
            continue
    all_issues.sort(key=lambda x: x.line_no)
    return all_issues


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Verilog Bug Finder (Free)", page_icon="🐛", layout="wide")

st.title("🐛 Verilog Bug Finder — Free MVP")
st.caption(
    "100% free, runs locally — no paid API key needed. "
    "Paste your Verilog code and get an instant rule-based review for logic bugs, "
    "timing/simulation risks, and style issues."
)

with st.expander("ℹ️ What does this check for?"):
    st.markdown(
        "- Blocking (`=`) assignments inside clocked `always` blocks\n"
        "- Non-blocking (`<=`) assignments inside combinational `always` blocks\n"
        "- `case` statements missing `default` / `if` missing `else` (latch risk)\n"
        "- Explicit sensitivity lists instead of `always @(*)`\n"
        "- Unsized numeric constants\n"
        "- Signals driven by more than one source\n"
        "- Clocked blocks with no reset signal\n"
        "- Very long combinational expressions (timing risk)"
    )

sample_code = """module counter(clk, rst, out);
input clk, rst;
output reg [3:0] out;

always @(posedge clk) begin
    if (rst)
        out = 0;
    else
        out = out + 1;
end

always @(a, b) begin
    if (sel)
        y = a;
end

endmodule
"""

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Paste your Verilog code")
    code_input = st.text_area(
        "Verilog source",
        value=st.session_state.get("code_input", ""),
        height=450,
        placeholder="module my_module(...);\n  ...\nendmodule",
        label_visibility="collapsed",
    )
    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
    with c2:
        if st.button("📋 Load Sample", use_container_width=True):
            st.session_state["code_input"] = sample_code
            st.rerun()

with col2:
    st.subheader("📊 Analysis Results")
    if run_btn:
        if not code_input.strip():
            st.warning("Please paste some Verilog code first.")
        else:
            issues = analyze_verilog(code_input)
            if not issues:
                st.success("✅ No issues found by the current rule set. Nice and clean!")
            else:
                bugs = [x for x in issues if x.severity == "Bug"]
                timing = [x for x in issues if x.severity == "Timing Risk"]
                style = [x for x in issues if x.severity == "Style"]

                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 Bugs", len(bugs))
                m2.metric("🟠 Timing Risks", len(timing))
                m3.metric("🟡 Style", len(style))

                icon = {"Bug": "🔴", "Timing Risk": "🟠", "Style": "🟡"}
                for issue in issues:
                    with st.container(border=True):
                        st.markdown(
                            f"{icon[issue.severity]} **{issue.severity}** — Line {issue.line_no}"
                        )
                        st.code(issue.line_text, language="verilog")
                        st.markdown(f"**Problem:** {issue.message}")
                        st.markdown(f"**Suggested fix:** {issue.suggestion}")
    else:
        st.info("Paste code on the left and click **Analyze Code** to see results here.")

st.divider()
st.caption(
    "This MVP uses free, offline static-analysis rules (no Claude API / paid calls). "
    "It's a first-pass linter, not a replacement for full simulation, formal verification, "
    "or static timing analysis (STA)."
)
