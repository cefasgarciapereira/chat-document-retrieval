import './App.css'
import { ChangeEvent, useState } from 'react'
import Modal, { ModalProps } from "./components/Modal"

interface ChatForm {
  question: string
  loading: Boolean
}

const initialFormState: ChatForm = {
  question: "",
  loading: false
}

interface Data {
  document: string,
  message: string,
  status: string
}

const initialModalState: ModalProps = {
  content: '',
  open: false
}

function App() {
  const [form, setForm] = useState<ChatForm>(initialFormState)
  const [data, setData] = useState<Data[]>([])
  const [modal, setModal] = useState<ModalProps>(initialModalState)

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const { name, value } = event.target
    setForm({ ...form, [name]: value })
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setForm({ ...form, loading: true })
    fetch(`http://127.0.0.1:5000/api/question?q=${form.question}`, { method: "GET" })
      .then(async (response) => {
        const newData = await response.json()
        setData([...data, newData])
        setForm({ ...form, loading: false })
      })
      .catch(error => {
        console.error(error)
        setForm({ ...form, loading: false })
      })
  }

  function loadingMessage() {
    const messages = ["hmmm...", "thinking", "searching", "wait", "hold on"]
    return form.loading ? messages[Math.floor(Math.random() * messages.length)] : "send"
  }

  function placeholderMessage() {
    const messages = [
      "What to do on PDV errors?",
      "Do you have a troubleshoot for orders without SKU?",
      "Where do I find the Broker emails?",
      "What does 'unexpected error on record to staging' means?",
    ]
    return `Ex: ${messages[Math.floor(Math.random() * messages.length)]}`
  }

  function openModal(content: string) {
    setModal({ content: content, open: true })
  }

  function closeModal() {
    setModal({ ...modal, open: false })
  }

  return (
    <>
      <div>
        <h1>Chat CI&T</h1>
      </div>
      <div className='message-box-container'>
        {
          data.map((item, index) => (
            <div className='message-box' key={index}>
              <div>
                <p style={{ whiteSpace: "pre-wrap" }}>
                  {item.message}
                </p>
              </div>
              <br />
              <button
                style={{ float: "right" }}
                onClick={() => openModal(item.document)}
              >
                more
              </button>
            </div>
          ))
        }
      </div>
      <div>
        <div className="form-card">
          <form onSubmit={handleSubmit} className='chat-form'>
            <input
              name="question"
              type="text"
              placeholder={placeholderMessage()}
              value={form.question}
              onChange={handleChange}
            />
            <button
              type="submit"
              disabled={!!form.loading}
            >
              {loadingMessage()}
            </button>
          </form>
        </div>

        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
        <p className="read-the-docs">
          Click on the Vite and React logos to learn more
        </p>
      </div>
      <Modal content={modal?.content} open={modal?.open} onClose={closeModal} />
    </>
  )
}

export default App
