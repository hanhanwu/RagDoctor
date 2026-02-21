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
  * `npx expo start --web` will start the web 