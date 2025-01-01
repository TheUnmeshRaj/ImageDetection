import React, { useState, useContext } from 'react';
import './ChatGPTStyle.css';
import axios from 'axios';
import { NoteContext } from './NoteContext';

function ChatGPTStyle() {
  const { chatHistory, setChatHistory } = useContext(NoteContext);
  const [inputText, setInputText] = useState('');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSubmit = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post('http://127.0.0.1:5002/api/ImageAi', formData);
      const newMessage = {
        type: 'image',
        query: image.name,
        result: response.data.result,
        treatment: formatTreatment(response.data.treatment),
        imageURL: URL.createObjectURL(image),
      };
      setChatHistory([...chatHistory, newMessage]);
      setImage(null);
      document.querySelector('.file-input').value = null;
    } catch (error) {
      console.error('Error fetching image result:', error);
      setError('Failed to fetch image result. Please try again.');
    }
    setLoading(false);
  };

  // Formatting function for treatment plans
  const formatTreatment = (treatment) => {
    const lines = treatment
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>') // Bold headings
      .replace(/\*(.+?)\*/g, '<em>$1</em>') // Italicize bullet points
      .split('\n')
      .map((line) => {
        if (line.startsWith('*')) {
          return `<li>${line.slice(1).trim()}</li>`; // Format list items
        }
        return `<p>${line.trim()}</p>`; // Format paragraphs
      })
      .join('');
    return `<div class="formatted-treatment">${lines}</div>`;
  };

  const renderChatHistory = () =>
    chatHistory.map((message, index) => (
      <div key={index} className="chat-bubble">
        <div className="query">
          {message.type === 'text' ? (
            <p>User: {message.query}</p>
          ) : (
            <>
              <p>User uploaded:</p>
              <img
                src={message.imageURL}
                alt={`Uploaded image: ${message.query}`}
                className="uploaded-img"
              />
            </>
          )}
        </div>
        <div className="response">
          <p>
            <strong>Result:</strong> {message.result}
          </p>
          <div
            className="treatment"
            dangerouslySetInnerHTML={{ __html: message.treatment }}
          ></div>
        </div>
      </div>
    ));

  return (
    <div className="chat-container">
      <h1>Skin Disease Detection AI</h1>
      {error && <p className="error-message">{error}</p>}
      {loading && <p className="loading">Processing...</p>}
      <div className="chat-history">{renderChatHistory()}</div>
      <div className="input-section">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
          className="file-input"
        />
        <button onClick={handleImageSubmit} disabled={loading || !image} className="send-btn">
          Upload Image
        </button>
      </div>
    </div>
  );
}

export default ChatGPTStyle;
