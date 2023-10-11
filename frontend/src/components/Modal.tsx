export interface ModalProps {
  open: boolean,
  content: string,
  onClose?: () => void
}

export default function Modal(props: ModalProps) {
  const { content, open, onClose } = props

  if (!open) return null

  return (
    <div className="modal">
      {content}
      <button onClick={onClose}>close</button>
    </div>
  )
}