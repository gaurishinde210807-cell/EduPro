"""
test_app.py
===========
Runs every page of the Streamlit dashboard headlessly and fails loudly on any exception.
Useful as a smoke test after re-running the pipeline on new data.

    python test_app.py
"""

from streamlit.testing.v1 import AppTest

PAGES = ["Overview", "Course demand predictor", "Revenue forecast",
         "Feature importance", "Category comparison", "Model performance"]

failures = 0
for page in PAGES:
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    if at.exception:
        print(f"FAIL  {page:28s} (initial load) -> {at.exception[0].value}")
        failures += 1
        continue
    at.sidebar.radio[0].set_value(page).run()
    if at.exception:
        print(f"FAIL  {page:28s} -> {at.exception[0].value}")
        failures += 1
    else:
        n_errors = len(at.error)
        status = "OK" if not n_errors else f"OK (with {n_errors} st.error blocks)"
        print(f"PASS  {page:28s} {status}")

print("\nAll pages rendered." if not failures else f"\n{failures} page(s) failed.")
raise SystemExit(1 if failures else 0)
