# RAG Doctor - Experiments Branch


## Learning Notes 🍀
### Financial Cost Notes 💸
* `from llama_index.embeddings.huggingface import HuggingFaceEmbedding` use lazy-loading so that it's used only when needed, don't import it at startup, otherwise the idel RAM will cumulate monthly and costs lots of money. It uses transformer model, which costs large RAM.

### Build Speed Up Notes 🚀
* If there's large packages to install, using docker is faster than Railway's default Railpack, cuz Dockerfile steps create layers and only layers that changed are rebuilt. So if requirements.txt stay the same, it's reusable with Docker build. Railpack treats every deployment as a fresh environment.
* `uv` install is faster than `pip` install as it uses Uses a precompiled wheel cache and optimized resolver, also skips unnecessary dependency checks.

### Prevent Multi-User Conflicts
* Because `rag_data` is global, only run data preprocessing once when multiple users selected same data simultiously 
* Added `asyncio.Lock` to allow only one user to run `/run-rags` each time to avoid Railway memory exhaustion or users' results contamination
* Groq could have 429 rate limit errors --> replace `_invoke_with_retry()` with `await chain.ainvoke()`

### How to Automatically Upload Files into Google Drive
* `pip install google-auth google-auth-oauthlib google-api-python-client openpyxl`
* Download credentials JSON into wuhan/.gcp/
  * Enable Google Drive API first
  * Go to APIs & Services → OAuth consent screen -> External -> Desktop App (type)
  * Download JSON before closing the window!
* Add your Google Drive login email as test user
  * Go to https://console.cloud.google.com/apis/credentials/consent
  * Audience -> Test users -> Add users
* Find your Google Drive folder ID through the folder's URL
* First time running `upload_to_google_drive()` it will generate `token.pickle` file after you finished authentication, later you don't need to do authentication any more


### Security
* Cloudflare's free domain is blocked by LinkedIn and other websites because too many fraud groups use those --> I purchased `hanhanwu.com` domain from Spaceship, then updated Cloudflare URL