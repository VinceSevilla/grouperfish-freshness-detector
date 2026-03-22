import { Prediction } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface PredictionDisplayProps {
  prediction: Prediction | null
  label: string
  detected: boolean
}

function getColorForClass(freshness: string): string {
  const normalized = freshness.toLowerCase()
  switch (normalized) {
    case 'fresh':
      return 'bg-green-200 dark:bg-green-900/50 text-green-900 dark:text-green-100 border-green-400 dark:border-green-600'
    case 'less_fresh':
      return 'bg-yellow-200 dark:bg-yellow-900/50 text-yellow-900 dark:text-yellow-100 border-yellow-400 dark:border-yellow-600'
    case 'starting_to_rot':
      return 'bg-orange-200 dark:bg-orange-900/50 text-orange-900 dark:text-orange-100 border-orange-400 dark:border-orange-600'
    case 'rotten':
      return 'bg-red-200 dark:bg-red-900/50 text-red-900 dark:text-red-100 border-red-400 dark:border-red-600'
    default:
      return 'bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-gray-100 border-gray-400 dark:border-gray-600'
  }
}

function formatClassName(className: string): string {
  return className
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}



export function PredictionDisplay({ prediction, label, detected }: PredictionDisplayProps) {
  if (!detected) {
    return (
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-base">{label}</CardTitle>
          <CardDescription>Not detected in image</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No {label.toLowerCase()} region was detected. Please upload or capture an image with a clear {label.toLowerCase()} region.
          </p>
        </CardContent>
      </Card>
    )
  }

  if (!prediction) {
    return (
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-base">{label}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Processing...</p>
        </CardContent>
      </Card>
    )
  }

  const colorClass = getColorForClass(prediction.class)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{label}</CardTitle>
        <CardDescription>Freshness Classification</CardDescription>
      </CardHeader>
      <CardContent>
        <div className={`p-4 rounded-lg border-2 ${colorClass} text-center`}>
          <p className="font-bold text-lg">{formatClassName(prediction.class)}</p>
        </div>
      </CardContent>
    </Card>
  )
}
