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

### Build Speed Up Notes 🚀
* If there's large packages to install, using docker is faster than Railway's default Railpack, cuz Dockerfile steps create layers and only layers that changed are rebuilt. So if requirements.txt stay the same, it's reusable with Docker build. Railpack treats every deployment as a fresh environment.
* `uv` install is faster than `pip` install as it uses Uses a precompiled wheel cache and optimized resolver, also skips unnecessary dependency checks.

## Test Railway Backend
* Deploy your new backend changes to Railway, once the service became live
* Open `https://hanhanchatbot-production.up.railway.app/{FastAPI function prefix}` to see results
  * For example: `https://hanhanchatbot-production.up.railway.app/debug/tables`