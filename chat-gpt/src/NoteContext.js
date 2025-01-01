import React, { createContext, useState } from 'react';

export const NoteContext = createContext();

const NoteContextProvider = ({ children }) => {
  const [chatHistory, setChatHistory] = useState([]);
  return (
    <NoteContext.Provider value={{ chatHistory, setChatHistory }}>
      {children}
    </NoteContext.Provider>
  );
};

export default NoteContextProvider;
