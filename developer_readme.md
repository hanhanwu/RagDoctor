# RAG Doctor - Developer Guidance

## Setup
* Install Node.js (latest LTS).
  * run `node -v` and `npm -v` to see whether both have output.
* Run `npm install -g expo-cli` to install the Expo CLI, a tool that helps developers create, develop, and manage React Native projects using the Expo framework.
* If "frontend" folder doesn't exist, creating it by running `npx create-expo-app frontend --template blank`, otherwise skip this step
  * folder name is "frontend" here
  * this should do `npm install` automatically
* `cd frontend`
  * `npx expo install react-dom react-native-web`


## Test Local Website
* `cd backend`
  * `python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000` to start FastAPI server
  * Backend console will show printed results
* `cd frontend`
  * `npx expo start --web` will start the web 🚀


## Railway Setup
* Make sure there's `requirements.txt` in your repo
  * In this case I have large packages like LlamIndex, using docker to build will be much faster than railway's default Railpack.
* Create `start.sh`, no need to add this to Railway's start command as Docker build will be automatically detected
* Create `Dockerfile` and `.dockerignore` files
* Push code changes and do railway deploy
* After a successful deployment, click the `Service` --> Click `Settings`
  * Under `Networking` --> Click `Generate Domain` --> port number can be 8080 --> Get `Railway URL`
  * You can test https://{Railway URL}/docs from browser, if it shows FastAPI page, then you're good
  * If you have multiple Railway projects, their domain can all be 8080, as long as they're separated deployments, cuz in Railway each project has its own container
  * Copy the generated domain to App.js as the value of `BACKEND_URL`, make sure you have `https://` before the URL!


## Publish Website
* Under folder "frontend/"
  * Run `npx expo export --platform web` to export web build, this should create a "dist/" folder
* In Cloudflare, search for `Worker & Pages` --> `Create New Application` --> Connect to github repo:
  * Root directory: `frontend`
  * Framework preset: leave blank
  * Build command: `npx expo export --platform web`
  * Build output directory: `dist`
* After successful deployment, this page is called as "ragdr", need to customize domain:
  * `Workers & Pages` --> `ragdr` --> `Custom domain` --> type `ragdr.hanhanwu.com` and wait till it's active
