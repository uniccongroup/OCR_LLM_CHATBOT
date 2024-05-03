import React, { useState } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [showReplaceConfirmation, setShowReplaceConfirmation] = useState(false);
  const [showChatInput, setShowChatInput] = useState(false);

  const handleImageChange = (event) => {
    const selectedImage = event.target.files[0];
    if (selectedImage) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(selectedImage);
    }
  };

  const handleAddImageClick = () => {
    if (image) {
      setShowReplaceConfirmation(true);
    } else {
      // If no image exists, trigger file input
      document.getElementById("imageInput").click();
    }
  };

  const handleReplaceConfirm = () => {
    setImage(null); // Clear existing image
    setShowChatInput(false);
    setShowReplaceConfirmation(false);
  };

  const handleReplaceCancel = () => {
    setShowReplaceConfirmation(false);
  };

  const handleStartChat = () => {
    // Clear existing image and show chat input
    setShowChatInput(true);
  };


  return (
    <div className="flex items-center justify-center h-screen bg-blue-500">
      <div className="flex w-[70%] h-[80%] justify-center bg-blue-600 rounded-lg">
        <div className="bg-white w-[30%] m-4 rounded-lg">
          <div className="bg-gray-100 rounded-lg h-[50%]">
            {image ? (
              <img
                src={image}
                alt="Selected Image"
                style={{ width: "100%", height: "100%" }}
                className="rounded-lg"
              />
            ) : (
              <div className="flex flex-row items-center justify-center h-full">
                insert image
                <span className="material-symbols-outlined">
                  add_photo_alternate
                </span>
              </div>
            )}
          </div>

          <input
            id="imageInput"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ display: "none" }}
          />
          <button
            onClick={handleAddImageClick}
            className="bg-gradient-to-r from-blue-600 to-blue-500  text-white font-semibold text-sm mx-2 my-4 py-2 px-4 rounded-md shadow-sm hover:shadow-xl transition duration-300"
          >
            {image ? "Replace Image" : "Add Image"}
          </button>

          {image ? (
            <button
              onClick={handleStartChat}
              className="bg-gradient-to-r from-gray-600 to-gray-700 text-white font-semibold text-sm mx-2 my-4 py-2 px-4 rounded-md shadow-sm hover:shadow-xl transition duration-300"
            >
              Start chat
            </button>
          ) : null}
        </div>
        <div className="bg-gradient-to-br from-gray-700 to-gray-600 w-[60%] rounded-lg m-4">
          {showChatInput ? (
            <div>
              <input
                type="text"
                placeholder="Type your message from picture here..."
                className="border border-gray-400 rounded-md p-2 m-4  w-[75%] focus:outline-none"
              />
              <button className="bg-blue-600 text-white font-semibold px-4 py-2 ml-2 rounded-md hover:bg-blue-700">
                Send
              </button>
            </div>
          ) : (
            <div className="bg-gray-800 h-[100%] p-8 rounded-lg shadow-lg">
              <h2 className="text-white mt-4 text-2xl text-center font-extrabold tracking-tight">
                An OCR you can chat with make life easy hurray
              </h2>
              <div className="flex justify-center mt-8 space-x-8">
                <div className="w-32 h-32 bg-blue-500 rounded-lg flex items-center justify-center text-white">
                  Better life
                </div>
                <div className="w-32 h-32 bg-green-500 rounded-lg flex items-center justify-center text-white">
                  Get insight
                </div>
                <div className="w-32 h-32 bg-yellow-500 rounded-lg flex items-center justify-center text-white">
                  How lovely
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Replaceinformation Magana */}
      {showReplaceConfirmation && (
        <div className="fixed top-0 left-0 w-full h-full bg-gray-900 bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-4 rounded-lg">
            <p>Do you want to replace the old image?</p>
            <div className="flex justify-end mt-4">
              <button
                onClick={handleReplaceConfirm}
                className="mr-2 px-4 py-2 bg-blue-500 text-white rounded-lg"
              >
                Yes
              </button>
              <button
                onClick={handleReplaceCancel}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg"
              >
                No
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
