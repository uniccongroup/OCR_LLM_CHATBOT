const messageBar = document.querySelector(".bar-wrapper input");
const sendBtn = document.querySelector(".bar-wrapper button");
const messageBox = document.querySelector(".message-box");

const API_URL = "http://localhost:8000/chat"; // Adjust the URL for your FastAPI server

sendBtn.onclick = function () {
  if (messageBar.value.length > 0) {
    const userTypedMessage = messageBar.value;
    messageBar.value = "";

    const userMessageHTML =
      `<div class="chat message">
        <img src="static/images/user.jpg">
        <span>${userTypedMessage}</span>
      </div>`;

    messageBox.insertAdjacentHTML("beforeend", userMessageHTML);

    setTimeout(() => {
      const requestOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: userTypedMessage // Send message content directly
        })
      };

      fetch(API_URL, requestOptions)
        .then(res => res.json())
        .then(data => {
          const chatBotResponse = data.response;
          renderChatBotResponse(chatBotResponse);
        })
        .catch(error => {
          renderChatBotResponse("Oops! An error occurred. Please try again.");
        });
    }, 100);
  }
};

function renderChatBotResponse(response) {
  const responseHTML =
    `<div class="chat response">
      <img src="static/images/chatbot.jpg">
      <span>${response}</span>
    </div>`;
  messageBox.insertAdjacentHTML("beforeend", responseHTML);
}
