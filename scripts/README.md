# Scripts

This folder contains scripts used in this project for miscellaneous tasks. The following table contains a description of each one:

---
| Script | Description |
|---------|-----------|
| generate_openapi.py | Export the OpenAPI specification of the API to `docs/docs/swagger_api.json`, with its version and update time. Run `python scripts/generate_openapi.py` after any change to the API (see section 5.9 of the main README) |
| generate_synthetic_data.py | Generate a synthetic dataframe to avoid dependency between model training code and pipeline code |
---