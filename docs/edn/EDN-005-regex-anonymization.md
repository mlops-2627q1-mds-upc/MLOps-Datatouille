**Engineering Decision Notebook (EDN) - Entry**

**Milestone:** M3: Model Building - Quality Assurance

**Activity / Topic (optional):** Data preprocessing and privacy.

**Decision Participants:** All team members.

**Decision:** Anonymize email addresses and international phone numbers in the cleaning stage with regular expressions, replacing them with the tokens `xxemail` and `xxphone`, instead of using a named entity recognition model based on BERT.

**Rationale:** The named entity recognition model required more computation than the training of the spam classifier itself, which contradicts the energy efficiency goals of the project. Regular expressions remove the most directly identifying contact details at a negligible cost. As a trade-off, local phone numbers without a country code and personal names are not detected. A Great Expectations rule verifies that no email address or phone number remains in the cleaned dataset.

**AI Involvement:** Select all that apply.

[x] No AI involvement

[ ] Information seeking

[ ] Alternative generation

[ ] Alternative assessment

[ ] Recommendation

[ ] Solution generation

[ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Other Evidence (optional):** `src/data/preprocess.py` (`EMAIL_PATTERN`, `PHONE_PATTERN`, `anonymize_text`), `tests/unit/test_anonymization.py`, and the Great Expectations suite defined in `src/data/gx_context_configuration.py` (`build_expectations`).
