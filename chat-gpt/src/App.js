import React from 'react';
import ChatGPTStyle from './ChatGPTStyle';
import NoteContextProvider from './NoteContext';

function App() {
  return (
    <NoteContextProvider>
      <div className="App">
        <ChatGPTStyle />
      </div>
    </NoteContextProvider>
  );
}

export default App;
