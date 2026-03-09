# Hanhan_Chatbot

## Setup
* Install Expo Go app from App Store / Google Play.
* On your computer, 
  * install Node.js (latest LTS).
  * Run node -v and npm -v to see whether both have output.
    * If they don't exist, ask Github Copilot how ot get them installed.
* In your Virtual Env,
  * Run `npm install -g expo-cli` to install the Expo CLI, a tool that helps developers create, develop, and manage React Native projects using the Expo framework.
* Create project folder by running `npx create-expo-app my_chatbot_frontend --template blank`
  * This will do `npm install` automatically
* `cd my_chatbot_frontend`
  * `npx expo install react-dom react-native-web`

## Test Local Website
* `cd chatbot_backend`
  * `python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000` to start FastAPI server
  * Backend console will show printed results
* `cd my_chatbot_frontend`
  * `npx expo start --web` will start the web 🚀

## Railway Setup
* Make sure there's requirements.txt in your repo
  * In this case I have large packages like LlamIndex, using docker to build will be much faster than railway's default Railpack.
* Create `start.sh`, no need to add this to Railway's start command as Docker build will be automatically detected
* Create `Dockerfile` and `.dockerignore` files
* Push code changes and do railway deploy
* After a successful deployment, click the `Service` --> Click `Settings`
  * Under `Networking` --> Click `Generate Domain` --> port number can be 8080 --> Get `Railway URL`
  * You can test https://{Railway URL}/docs from browser, if it shows FastAPI page, then you're good
  * If you have multiple Railway projects, their domain can all be 8080, as long as they're separated deployments, cuz in Railway each project has its own container
  * Copy the generated domain to App.js as the value of `BACKEND_URL`, make sure you have `https://` before the URL!


## Test Railway Backend
* Deploy your new backend changes to Railway, once the service became live
* Open `https://hanhanchatbot-production.up.railway.app/{FastAPI function prefix}` to see results
  * For example: `https://hanhanchatbot-production.up.railway.app/debug/tables`

## Publish Website
* Under folder "my_chatbot_frontend/"
  * Run `npx expo export --platform web` to export web build, this should create a "dist/" folder
* In Cloudflare, search for "Worker & Pages" --> "Create New Application"
  * Manually upload "dist/" folder here --> Click "Deploy"
  * After deployment successfully, can connect to Github repo
    * Then under "Settings" --> "Build Configuration":
      * Build Command: `npx expo export --platform web`
      * Root directory: `my_chatbot_frontend`


## Learning Notes 🍀
### Build Speed Up Notes 🚀
* If there's large packages to install, using docker is faster than Railway's default Railpack, cuz Dockerfile steps create layers and only layers that changed are rebuilt. So if requirements.txt stay the same, it's reusable with Docker build. Railpack treats every deployment as a fresh environment.
* `uv` install is faster than `pip` install as it uses Uses a precompiled wheel cache and optimized resolver, also skips unnecessary dependency checks.

### Prevent Multi-User Conflicts
* Because `rag_data` is global, only run data preprocessing once when multiple users selected same data simultiously 
* Added `asyncio.Lock` to allow only one user to run `/run-rags` each time to avoid Railway memory exhaustion or users' results contamination
  * 🍄 TO-DO: change to Job queue for multi-users, rather than having users re-run manually after lock released
* Groq could have 429 rate limit errors --> replace `_invoke_with_retry()` with `await chain.ainvoke()`

### Security
* Cloudflare's free domain is blocked by LinkedIn and other websites because too many fraud groups use those --> I purchased `hanhanwu.com` domain from Spaceship, then updated Cloudflare URL