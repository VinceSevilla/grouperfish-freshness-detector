import { AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Modal } from '@/components/ui/modal'

interface PredictionWarningModalProps {
  isOpen: boolean
  onClose: () => void
  onRetry: () => void
  missingPredictions: {
    eyeMissing: boolean
    gillMissing: boolean
  }
}

export function PredictionWarningModal({ isOpen, onClose, onRetry, missingPredictions }: PredictionWarningModalProps) {
  const { eyeMissing, gillMissing } = missingPredictions
  
  let title = ''
  let message = ''
  let suggestions: string[] = []

  if (eyeMissing && gillMissing) {
    title = 'No Predictions Available'
    message = 'Neither the eyes nor the gills could be detected or analyzed in the image.'
    suggestions = [
      'Ensure the fish is clearly visible in the image',
      'Make sure both the eyes and gills are in the frame',
      'Position the fish so that the eyes and lateral view of gills are visible',
      'Ensure adequate lighting for the eyes and gill area',
      'Remove any obstructions blocking the eyes or gills',
      'Try capturing from a different angle',
      'Upload a higher quality image'
    ]
  } else if (eyeMissing) {
    title = 'Eye Prediction Not Available'
    message = 'The eyes could not be detected or analyzed in the image, but the gill analysis is available.'
    suggestions = [
      'Ensure the fish eyes are clearly visible',
      'Make sure the head of the fish is clearly in frame',
      'Provide adequate lighting on the eyes',
      'Position the fish to show the eyes without obstruction',
      'Ensure the eye area is in focus',
      'Try a closer shot of the head area'
    ]
  } else if (gillMissing) {
    title = 'Gill Prediction Not Available'
    message = 'The gills could not be detected or analyzed in the image, but the eye analysis is available.'
    suggestions = [
      'Ensure the gills are clearly visible',
      'Show the lateral (side) view of the fish where gills are located',
      'Provide adequate lighting on the gill area',
      'Remove any obstructions covering the gills',
      'Make sure the gill area is in focus',
      'Try capturing the fish from a side angle to show gills clearly'
    ]
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 pt-0.5">
            <AlertCircle className="h-6 w-6 text-orange-600" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{message}</p>
          </div>
        </div>

        {/* Suggestions */}
        <div className="space-y-3">
          <h3 className="font-medium text-gray-900 dark:text-white">What you can try:</h3>
          <ul className="space-y-2">
            {suggestions.map((suggestion, index) => (
              <li key={index} className="flex gap-3 text-sm text-gray-700 dark:text-gray-300">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center text-xs font-medium text-orange-700 dark:text-orange-400">
                  {index + 1}
                </span>
                <span>{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Actions */}
        <div className="flex gap-3 justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button onClick={onRetry} className="bg-orange-600 hover:bg-orange-700">
            Try Again
          </Button>
        </div>
      </div>
    </Modal>
  )
}
