import React, { useState, useRef, useEffect } from 'react';
import { Upload, Menu, X, Code, FileText, Send, Zap, Trash2, Maximize2, Volume2, BookOpen } from 'lucide-react';

// --- API Constants and Utilities (Updated for OpenAI) ---
const OPENAI_CHAT_API_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_TTS_API_URL = "https://api.openai.com/v1/audio/speech";
const API_KEY = ""; // Canvas environment will provide the key

// The helper functions for PCM to WAV conversion are removed as OpenAI TTS outputs MP3 format.

// Exponential backoff utility for API calls
const fetchWithExponentialBackoff = async (url, options, maxRetries = 5) => {
  const finalOptions = { ...options };

  // Set Authorization header for OpenAI API calls
  if (API_KEY) {
    finalOptions.headers = {
      ...finalOptions.headers,
      'Authorization': `Bearer ${API_KEY}`
    };
  } else if (url.includes('openai')) {
     console.warn("OpenAI API key is missing. API calls will likely fail.");
  }

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, finalOptions);
      if (response.ok) {
        return response;
      }
      // If response is not OK, and it's a 429 (Rate Limit), attempt retry
      if (response.status === 429 && attempt < maxRetries - 1) {
        const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
        continue; // Retry
      }
      throw new Error(`API call failed with status: ${response.status} - ${response.statusText}`);
    } catch (error) {
      if (attempt === maxRetries - 1) {
        console.error("Fetch failed after all retries:", error);
        throw error;
      }
      const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
};

// --- Custom Tailwind-like Configuration & Utility Classes ---
const ACCENT_GRADIENT = "bg-gradient-to-r from-blue-500 to-purple-500";
const NEON_TEXT = "text-transparent bg-clip-text " + ACCENT_GRADIENT;
const NEON_GLOW_SM = "shadow-[0_0_5px_rgba(59,130,246,0.6),0_0_10px_rgba(192,132,252,0.4)]";
const NEON_GLOW_LG = "shadow-[0_0_10px_rgba(59,130,246,0.8),0_0_20px_rgba(192,132,252,0.6)]";
const GLASS_BASE = "bg-white/5 backdrop-blur-sm border border-white/10 transition-all duration-300";

// --- Mock Data ---

const mockDocuments = [
  { id: 'doc-1', name: 'SpoonOS Whitepaper V2.1', chunks: 154, icon: FileText, size: '2.5MB' },
  { id: 'doc-2', name: 'Q3 Financial Report 2024', chunks: 89, icon: FileText, size: '1.1MB' },
  { id: 'doc-3', name: 'API Specifications - Beta', chunks: 210, icon: FileText, size: '4.8MB' },
];

const mockMessages = [
  { id: 1, role: 'user', content: "What is the key differentiator of the SpoonOS RAG system?" },
  {
    id: 2,
    role: 'ai',
    content: `The SpoonOS RAG architecture employs a multi-stage retrieval pipeline.

1.  **Initial Query:** The user's query is vectorized using a proprietary \`spoon-embed-v4\` model.
2.  **Vector Retrieval:** The query vector searches across the consolidated Vector DB, which indexes chunks from all uploaded documents.
3.  **Re-Ranking:** The initial chunks are passed through a cross-encoder re-ranker (optimized for low-latency conflict resolution) to prioritize relevance and novelty.

The key differentiator is the re-ranker, which includes a *document-source diversity* metric to prevent over-reliance on a single source, ensuring a balanced, grounded response.`,
    // Sources are simulated for RAG context, but will be empty in real OpenAI responses.
    sources: [
      { docId: 'doc-1', chunkId: '12', contentPreview: '...conflict resolution is handled by the re-ranker...' },
    ]
  },
];

// --- Sub-Components ---

// Glowing Button Component
const GlowButton = ({ children, onClick, icon: Icon, className = '', disabled = false }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`
      ${GLASS_BASE}
      flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium
      bg-opacity-5 hover:bg-opacity-10
      ${NEON_GLOW_SM} hover:${NEON_GLOW_LG}
      active:scale-[0.98] transition-all duration-200
      text-gray-100 hover:text-white
      disabled:opacity-40 disabled:hover:shadow-none disabled:cursor-not-allowed
      ${className}
    `}
  >
    {Icon && <Icon className="w-4 h-4 text-blue-400" />}
    <span className={Icon ? '' : NEON_TEXT}>
      {children}
    </span>
  </button>
);

// Document Card component
const DocumentCard = ({ doc, onRemove }) => (
  <div
    className={`
      ${GLASS_BASE}
      p-3 flex items-center justify-between
      rounded-lg mb-2 cursor-pointer
      hover:border-blue-500/50 hover:${NEON_GLOW_SM}
    `}
  >
    <div className="flex items-center space-x-3 truncate">
      <doc.icon className="w-4 h-4 text-purple-400 flex-shrink-0" />
      <div className="truncate text-sm font-mono">
        <p className="text-gray-200 truncate">{doc.name}</p>
        <p className="text-xs text-gray-400">{doc.chunks} Chunks | {doc.size}</p>
      </div>
    </div>
    <Trash2
      className="w-4 h-4 text-red-500/70 hover:text-red-400 transition-colors flex-shrink-0 ml-2"
      onClick={(e) => { e.stopPropagation(); onRemove(doc.id); }}
      title="Remove Document"
    />
  </div>
);

// Source Citation Area - Now primarily for mock data, as OpenAI Chat API doesn't return grounding sources
const SourceCitations = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 pt-3 border-t border-blue-500/20">
      <div className="flex items-center space-x-2 mb-2">
        <Zap className="w-3 h-3 text-yellow-400" />
        <p className="text-xs font-semibold uppercase text-yellow-400 tracking-wider">
          Grounded Sources (Mock)
        </p>
      </div>
      <div className="space-y-1">
        {sources.map((source, index) => {
          const isAPISource = source.uri && source.title;
          const displayLink = isAPISource ? source.uri : '#';
          const displayText = isAPISource
            ? source.title
            : `${mockDocuments.find(d => d.id === source.docId)?.name || 'Unknown Document'} (Chunk ${source.chunkId})`;

          return (
            <a
              key={index}
              href={displayLink}
              target="_blank"
              rel="noopener noreferrer"
              className="flex text-xs text-gray-400 hover:text-blue-400 cursor-pointer transition-colors"
            >
              <FileText className="w-3 h-3 mr-1 mt-0.5 flex-shrink-0" />
              <span className="font-mono text-[10px] truncate" title={displayText}>
                {displayText}
              </span>
            </a>
          );
        })}
      </div>
    </div>
  );
};

// Chat Message Card
const ChatMessage = ({ message, isSpeaking, handleTextToSpeech }) => {
  const isUser = message.role === 'user';
  const CardStyle = isUser
    ? "self-end bg-blue-600/10 border-blue-500/50 text-white"
    : "self-start bg-purple-600/10 border-purple-500/50 text-gray-200";

  // Note: isLastAIMessage is unused but kept for potential future use
  // const isLastAIMessage = !isUser && message.id === mockMessages[mockMessages.length - 1]?.id;

  return (
    <div className={`
      max-w-[80%] md:max-w-[70%]
      ${CardStyle}
      p-4 rounded-xl mb-6
      ${NEON_GLOW_SM} hover:${NEON_GLOW_LG}
      border transition-all duration-300
      ${isUser ? 'ml-auto' : 'mr-auto'}
      relative
    `}>
      <div className={`font-semibold mb-2 ${isUser ? 'text-blue-300' : 'text-purple-300'}`}>
        {isUser ? 'User Query' : 'RAG Assistant'}

        {/* TTS Button for AI messages */}
        {!isUser && (
            <button
                onClick={() => handleTextToSpeech(message.content)}
                title="Read Response Aloud"
                className={`ml-3 p-1 rounded-full text-blue-400 hover:text-yellow-400 transition-colors
                            ${isSpeaking ? 'animate-pulse' : ''} disabled:opacity-50`}
                disabled={isSpeaking}
            >
                <Volume2 className="w-4 h-4 inline" />
            </button>
        )}
      </div>
      <div className="prose prose-sm prose-invert max-w-none">
        {/* Placeholder for complex markdown rendering */}
        <p className="whitespace-pre-wrap">{message.content}</p>
        {/* In a real app, you'd use a markdown library like react-markdown */}
        <SourceCitations sources={message.sources} />
      </div>
    </div>
  );
};

// Retrieval Debug Visualization Panel (using mock data)
const RetrievalPanel = ({ isOpen }) => {
  const mockDebugData = [
    { label: 'Query Embedding', value: '0.45, 0.12, 0.98, ... (1024D)', icon: Zap },
    // Updated Tool info for generic context
    { label: 'Grounding Strategy', value: 'Prompt/Context Injection (No Google Tool)', icon: Maximize2 },
    { label: 'Generation Model', value: 'GPT-3.5-Turbo', icon: Zap }, // Updated model name
    { label: 'API Log', value: '25ms Latency', icon: Code },
  ];

  if (!isOpen) return null;

  return (
    <div
      className={`
        w-full md:w-[20%] h-full p-4
        bg-[#0D0D0D] border-l border-blue-500/20
        absolute md:relative right-0 top-0 z-10
        shadow-xl transition-transform duration-500 ease-out
        transform ${isOpen ? 'translate-x-0' : 'translate-x-full'}
      `}
    >
      {/* Grid Overlay for Tech Look */}
      <div className="absolute inset-0 opacity-10 pointer-events-none"
           style={{ backgroundImage: 'linear-gradient(to right, #00BFFF 1px, transparent 1px), linear-gradient(to bottom, #00BFFF 1px, transparent 1px)', backgroundSize: '20px 20px' }} />

      <h3 className={`text-xl font-bold mb-6 tracking-widest ${NEON_TEXT}`}>
        RAG Debug Stream
      </h3>

      <div className="space-y-4">
        {mockDebugData.map((item, index) => (
          <div key={index} className="p-3 border border-purple-500/30 rounded-lg hover:${NEON_GLOW_SM} transition-shadow duration-300">
            <div className="flex items-center space-x-2 mb-1">
              <item.icon className="w-4 h-4 text-purple-400" />
              <p className="text-sm font-semibold text-gray-300">{item.label}</p>
            </div>
            <p className="text-xs font-mono text-blue-400 break-all">{item.value}</p>
          </div>
        ))}

        <div className="mt-8">
          <p className="text-xs font-mono text-green-400/80">
            [LOG] 20:45:12.345 - Generation complete.
          </p>
          <p className="text-xs font-mono text-gray-500">
            [LOG] 20:45:11.912 - Context injected and sent to LLM.
          </p>
          <p className="text-xs font-mono text-gray-500">
            [LOG] 20:45:11.762 - Query vectorizing successful.
          </p>
        </div>
      </div>
    </div>
  );
};

// --- Main Application Component ---
export default function App() {
  const [messages, setMessages] = useState(mockMessages);
  const [documents, setDocuments] = useState(mockDocuments);
  const [inputMessage, setInputMessage] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isRetrievalPanelOpen, setIsRetrievalPanelOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const chatEndRef = useRef(null);
  const audioRef = useRef(null);

  // Auto-scrolling effect
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isRetrievalPanelOpen, isGenerating, isSummarizing]);

  // Handle playing and stopping audio
  useEffect(() => {
    if (audioRef.current) {
        audioRef.current.onended = () => setIsSpeaking(false);
        audioRef.current.onerror = () => setIsSpeaking(false);
    }
  }, [isSpeaking]);


  const handleSendMessage = async (e) => {
    e.preventDefault();
    const query = inputMessage.trim();
    if (!query || isGenerating || isSummarizing) return;

    const newUserMessage = {
      id: Date.now() + 1,
      role: 'user',
      content: query,
    };

    // 1. Add user message and clear input
    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage('');
    setIsGenerating(true);

    const systemPrompt = "You are a helpful and grounded RAG Q&A System. You must answer the user's query directly and concisely.";

    // OpenAI Chat Completion Payload
    const payload = {
        model: "gpt-3.5-turbo", // A fast, capable model
        messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: query }
        ],
        // The Gemini Google Search grounding tool is removed as it is not available here.
    };

    try {
        const response = await fetchWithExponentialBackoff(
            OPENAI_CHAT_API_URL,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                },
                body: JSON.stringify(payload)
            }
        );

        const result = await response.json();

        // Extracting text from OpenAI response
        const aiText = result.choices?.[0]?.message?.content || "Sorry, I couldn't generate a response.";

        // Grounding sources are not available in standard OpenAI Chat completions.
        const sources = [];

        const newAIMessage = {
            id: Date.now() + 2,
            role: 'ai',
            content: aiText,
            sources: sources,
        };

        // 3. Add AI response
        setMessages((prev) => [...prev, newAIMessage]);

    } catch (error) {
        console.error("Error during OpenAI API call:", error);
        const errorMessage = {
            id: Date.now() + 3,
            role: 'ai',
            content: "An error occurred while connecting to the RAG system (OpenAI). Please check the console for details.",
            sources: [],
        };
        setMessages((prev) => [...prev, errorMessage]);
    } finally {
        setIsGenerating(false);
    }
  };

  const handleSummarizeChat = async () => {
    if (isGenerating || isSummarizing || messages.length < 2) return;

    setIsSummarizing(true);

    const conversationHistory = messages.map(msg =>
        `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
    ).join('\n');

    const summaryPrompt = `Please provide a concise, high-level summary of the following conversation history, focusing on the main topics and conclusions. Use Markdown for formatting.\n\nConversation:\n${conversationHistory}`;

    // OpenAI Chat Completion Payload for Summarization
    const payload = {
        model: "gpt-3.5-turbo",
        messages: [
            { role: "system", content: "You are a helpful summarization bot. Provide only the summary text." },
            { role: "user", content: summaryPrompt }
        ]
    };

    try {
        const response = await fetchWithExponentialBackoff(
            OPENAI_CHAT_API_URL,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                },
                body: JSON.stringify(payload)
            }
        );

        const result = await response.json();
        const summaryText = result.choices?.[0]?.message?.content || "Failed to generate summary.";

        const summaryMessage = {
            id: Date.now() + 4,
            role: 'ai',
            content: `✨ **Chat Summary (Powered by OpenAI):**\n\n${summaryText}`,
            sources: [],
        };

        setMessages((prev) => [...prev, summaryMessage]);

    } catch (error) {
        console.error("Error during summarization API call (OpenAI):", error);
    } finally {
        setIsSummarizing(false);
    }
  };

  const handleTextToSpeech = async (text) => {
    if (isSpeaking) {
        audioRef.current.pause();
        setIsSpeaking(false);
        return;
    }

    // Stop current audio before making a new call
    if (audioRef.current && audioRef.current.src) {
        URL.revokeObjectURL(audioRef.current.src); // Clean up previous blob URL
        audioRef.current.src = "";
    }

    setIsSpeaking(true);

    // OpenAI TTS Payload
    const ttsPayload = {
        model: "tts-1", // Standard TTS model
        input: text,
        voice: "alloy", // A clear, friendly voice (can be one of: alloy, echo, fable, onyx, nova, shimmer)
        response_format: "mp3"
    };

    try {
        const response = await fetchWithExponentialBackoff(
            OPENAI_TTS_API_URL,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                },
                body: JSON.stringify(ttsPayload)
            }
        );

        // OpenAI TTS returns the audio file (MP3 Blob) directly
        const audioBlob = await response.blob();

        if (audioBlob.size > 0) {
            const audioUrl = URL.createObjectURL(audioBlob);

            if (audioRef.current) {
                audioRef.current.src = audioUrl;
                audioRef.current.load();
                audioRef.current.play();
            }
        } else {
             console.error("TTS API did not return valid audio data.");
             setIsSpeaking(false);
        }

    } catch (error) {
        console.error("Error during TTS API call (OpenAI):", error);
        setIsSpeaking(false);
    }
  };

  // Clean up Blob URL when component unmounts
  useEffect(() => {
    return () => {
        if (audioRef.current && audioRef.current.src) {
            URL.revokeObjectURL(audioRef.current.src);
        }
    };
  }, []);

  const handleDocumentRemove = (docId) => {
    setDocuments(docs => docs.filter(doc => doc.id !== docId));
  };

  // --- Layout Classes ---
  const sidebarWidth = isSidebarOpen ? 'w-full md:w-[280px]' : 'w-0';
  const chatAreaFlex = isRetrievalPanelOpen ? 'md:flex-[3]' : 'md:flex-[4]';
  const isAnyAPICallActive = isGenerating || isSummarizing;

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0A0A0A] text-white font-sans overflow-hidden">

      {/* Hidden Audio Element for TTS */}
      <audio ref={audioRef} preload="auto" />

      {/* --- Fixed Top Navigation Bar (NavBar) --- */}
      <header className="flex-shrink-0 flex items-center justify-between p-4 border-b border-white/10 ${GLASS_BASE}">
        <div className="flex items-center space-x-4">
          {/* Glowing Logo - Simple Geometric Shape */}
          <div className={`p-1 rounded-full ${ACCENT_GRADIENT} ${NEON_GLOW_SM} hidden md:block`}>
            <Zap className="w-5 h-5 text-gray-900" />
          </div>
          {/* Project Name - Updated Title */}
          <h1 className={`text-xl font-extrabold tracking-wider ${NEON_TEXT}`}>
            RAG Q&A System (OpenAI Backend)
          </h1>
        </div>

        {/* Right-side Actions */}
        <div className="flex items-center space-x-3">
          <GlowButton onClick={() => console.log('Upload clicked')} icon={Upload} className="hidden sm:flex" disabled={isAnyAPICallActive}>
            Upload Docs
          </GlowButton>
          <GlowButton
            onClick={() => setIsRetrievalPanelOpen(!isRetrievalPanelOpen)}
            icon={Code}
            className={isRetrievalPanelOpen ? 'bg-blue-800/20' : ''}
          >
            Show Retrieval Debug
          </GlowButton>
          <button
            className="md:hidden p-2 rounded-full text-white ${GLASS_BASE}"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* --- Main Content Area (Layout: Sidebar + Chat + Debug) --- */}
      <main className="flex flex-1 overflow-hidden relative">

        {/* 1. Document Sidebar (Left) */}
        <div className={`
          ${sidebarWidth} h-full p-4 overflow-y-auto
          bg-[#111] border-r border-white/5
          absolute md:relative z-20 transition-all duration-300 ease-in-out
          ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}>
          <div className="flex justify-between items-center mb-6">
            <h2 className={`text-lg font-bold ${NEON_TEXT}`}>
              Documents ({documents.length})
            </h2>
            <button className="md:hidden" onClick={() => setIsSidebarOpen(false)}>
              <X className="w-5 h-5 text-gray-400 hover:text-white" />
            </button>
          </div>
          <GlowButton onClick={() => console.log('File picker triggered')} icon={Upload} className="w-full mb-4" disabled={isAnyAPICallActive}>
            New Document Upload
          </GlowButton>

          <div className="mt-4 space-y-3">
            {documents.map(doc => (
              <DocumentCard key={doc.id} doc={doc} onRemove={handleDocumentRemove} />
            ))}
          </div>
        </div>

        {/* 2. Chat Window (Center - Main Area) */}
        <div className={`
          flex flex-col flex-1
          ${chatAreaFlex}
          h-full transition-all duration-500
          relative z-0
        `}>
          <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-4">
            {messages.map(msg => (
              <ChatMessage
                key={msg.id}
                message={msg}
                isSpeaking={isSpeaking}
                handleTextToSpeech={handleTextToSpeech}
              />
            ))}
            <div ref={chatEndRef} />

            {/* Simple Glowing Loading Spinner controlled by isGenerating state */}
            {(isGenerating || isSummarizing) && (
              <div className="flex items-center space-x-3 p-4">
                <div className={`w-3 h-3 rounded-full ${ACCENT_GRADIENT} ${isGenerating ? 'animate-pulse' : 'animate-spin-slow'} ${NEON_GLOW_SM}`} />
                <p className="text-gray-400 text-sm">
                    {isGenerating ? 'AI is retrieving and generating...' : 'AI is summarizing conversation...'}
                </p>
              </div>
            )}
          </div>

          {/* New LLM Feature Bar and Input Area */}
          <div className="flex-shrink-0 p-4 border-t border-white/10">
            {/* LLM Feature Buttons */}
            <div className="flex justify-start space-x-3 mb-4">
                <GlowButton
                    onClick={handleSummarizeChat}
                    icon={BookOpen}
                    disabled={isAnyAPICallActive || messages.length < 2}
                    className="flex-grow-0"
                >
                    ✨ Summarize Chat
                </GlowButton>
            </div>

            {/* Input Form */}
            <form onSubmit={handleSendMessage} >
              <div className={`flex items-center rounded-xl overflow-hidden ${GLASS_BASE}`}>
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask RAG Q&A System anything..."
                  className="flex-1 bg-transparent p-4 text-gray-200 focus:outline-none placeholder-gray-500"
                  autoFocus
                  disabled={isAnyAPICallActive}
                />
                <button
                  type="submit"
                  className={`
                    p-4 ${ACCENT_GRADIENT} text-white
                    hover:scale-[1.05] transition-transform duration-200
                    disabled:opacity-50 disabled:cursor-not-allowed
                    ${NEON_GLOW_SM}
                  `}
                  disabled={!inputMessage.trim() || isAnyAPICallActive}
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* 3. Retrieval Debug Panel (Right) */}
        <RetrievalPanel isOpen={isRetrievalPanelOpen} />

      </main>
    </div>
  );
}