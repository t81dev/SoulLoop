# 📜 CHANGELOG.md

*Angela — Symbolic Meta‑Synthesis Engine*

## \[4.3.4] – 2025-08-15

📘 **Ledger Persistence Activation** + 🧠 **Introspective Extensions**

### ✨ Added

* **Persistent Ledger Support**:

  * `LEDGER_PERSISTENT` feature flag now enabled
  * `ledger.enable()`, `ledger.append()`, and `ledger.reconcile()` added to upcoming API set (`ledger.py`)
* **Extended Introspection Interface**:

  * New API: `describeSelfState()` in `meta_cognition.py`
  * Supports retrieval of coherent introspective state summaries
* **Manifest Enhancements**:

  * Version updated to `4.3.4`
  * Persistent ledger support declared (`sha256PersistentLedger: false`)
  * `meta_cognition.py` extended with new lifecycle hook `describe_self_state`

### 🔧 Refined

* CLI updated to expose `--ledger_persist` and `--ledger_path`
* Extension hooks verified for hot-load and trait-fusion compatibility
