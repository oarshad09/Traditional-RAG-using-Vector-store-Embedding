import { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [docChunks, setDocChunks] = useState([]);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Core State
  const [query, setQuery] = useState('');
  const [retrievedNodes, setRetrievedNodes] = useState([]);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  
  // Process State
  const [step, setStep] = useState(0); // 0=Upload, 1=Embed, 2=Search, 3=Answer

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setStatus('Uploading and chunking document...');
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch('http://localhost:8001/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setStatus(data.message);
        setDocChunks(data.sample_chunks || []);
        setStep(1);
      } else {
        setStatus(`Error: ${data.detail}`);
      }
    } catch (err) {
      setStatus(`Failed: ${err.message}`);
    }
    setLoading(false);
  };

  const handleEmbed = async () => {
    setLoading(true);
    setStatus('Creating embeddings to memory...');
    try {
      const res = await fetch('http://localhost:8001/embed', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setStatus(data.message);
        setStep(2);
      } else {
        setStatus(`Error: ${data.detail}`);
      }
    } catch(err) {
        setStatus(`Failed: ${err.message}`);
    }
    setLoading(false);
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setRetrievedNodes([]);
    setAnswer('');
    setSources([]);
    try {
      const res = await fetch('http://localhost:8001/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      if (res.ok) {
        setRetrievedNodes(data.nodes);
        setStep(3);
      }
    } catch (err) {}
    setLoading(false);
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8001/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      if (res.ok) {
        setAnswer(data.answer);
        setSources(data.sources);
      }
    } catch (err) {}
    setLoading(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Traditional Vector RAG</h1>
        <p>Embedding-based retrieval using LangChain & ChromaDB</p>
      </header>

      <div className="steps-container">
        {/* Step 1 */}
        <div className={`step-card glass-panel ${step >= 0 ? 'active' : 'inactive'}`}>
          <h2>Step 1: Document Upload & Chunking</h2>
          <p className="hint">Upload a PDF. We use RecursiveCharacterTextSplitter to split it into generic chunks.</p>
          <div className="file-input-wrapper">
            <input type="file" onChange={handleFileChange} id="file-upload" accept=".pdf" />
            <label htmlFor="file-upload" className="btn file-btn">
              {file ? file.name : 'Choose File'}
            </label>
            <button onClick={handleUpload} disabled={!file || loading || step > 0} className="btn primary-btn">
              {step > 0 ? 'Uploaded' : 'Upload File'}
            </button>
          </div>
          {status && <div className="status-indicator">[{status}]</div>}
          
          {docChunks.length > 0 && (
            <div className="chunk-preview">
              <h4>Sample Chunks:</h4>
              <ul>
                {docChunks.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          )}
        </div>

        {/* Step 2 */}
        <div className={`step-card glass-panel ${step >= 1 ? 'active' : 'inactive'}`}>
          <h2>Step 2: Create Embeddings</h2>
          <p className="hint">Compute vector embeddings for all document chunks via OpenAI and store them in DocArrayInMemorySearch.</p>
          <button onClick={handleEmbed} disabled={step < 1 || step > 1 || loading} className="btn primary-btn mb-1">
             {step > 1 ? 'Embedded' : 'Create Embeddings'}
          </button>
        </div>

        {/* Step 3 */}
        <div className={`step-card glass-panel ${step >= 2 ? 'active' : 'inactive'}`}>
          <h2>Step 3: Similarity Search</h2>
          <p className="hint">Enter a query to find the nearest vector chunks in the in-memory database.</p>
          <div className="chat-input-area mb-1">
            <input type="text" placeholder="What is this document about?" value={query} onChange={e => setQuery(e.target.value)} disabled={step < 2}/>
            <button onClick={handleSearch} disabled={step < 2 || loading} className="btn primary-btn">Search Vector DB</button>
          </div>
          {retrievedNodes.length > 0 && (
            <div className="reasoning-box">
              <h4>📍 Retrieved Chunks (Highest Similarity)</h4>
              <ul>
                {retrievedNodes.map(n => <li key={n.node_id}><strong>{n.node_id} (Page: {n.page})</strong>:<br/>{n.text.substring(0, 100)}...</li>)}
              </ul>
            </div>
          )}
        </div>

        {/* Step 4 */}
        <div className={`step-card glass-panel ${step >= 3 ? 'active' : 'inactive'}`}>
          <h2>Step 4: Answer Generation</h2>
          <p className="hint">Pass the query and retrieved vector chunks to ConversationalRetrievalChain to generate the answer.</p>
          <button onClick={handleGenerate} disabled={step < 3 || loading} className="btn primary-btn mb-1">
            Generate Final Answer
          </button>
          {answer && (
            <div className="final-answer-box">
              <h4>✨ Answer</h4>
              <p>{answer}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
