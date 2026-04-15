import { Prediction } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface PredictionDisplayProps {
  prediction: Prediction | null
  label: string
  detected: boolean
  eyePrediction?: Prediction | null
  gillPrediction?: Prediction | null
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



export function PredictionDisplay({ prediction, label, detected, eyePrediction = null, gillPrediction = null }: PredictionDisplayProps) {
  if (!detected) {
    return (
      <Card className="bg-muted/50">
        <CardHeader className="p-3 sm:p-6">
          <CardTitle className="text-sm sm:text-base">{label}</CardTitle>
          <CardDescription className="text-xs sm:text-sm">Not detected in image</CardDescription>
        </CardHeader>
        <CardContent className="p-3 sm:p-6 pt-0 sm:pt-0">
          <p className="text-xs sm:text-sm text-muted-foreground">
            No {label.toLowerCase()} region was detected. Please upload or capture an image with a clear {label.toLowerCase()} region.
          </p>
        </CardContent>
      </Card>
    )
  }

  if (!prediction) {
    return (
      <Card className="bg-muted/50">
        <CardHeader className="p-3 sm:p-6">
          <CardTitle className="text-sm sm:text-base">{label}</CardTitle>
        </CardHeader>
        <CardContent className="p-3 sm:p-6 pt-0 sm:pt-0">
          <p className="text-xs sm:text-sm text-muted-foreground">Processing...</p>
        </CardContent>
      </Card>
    )
  }

  const colorClass = getColorForClass(prediction.class)

  return (
    <Card>
      <CardHeader className="p-3 sm:p-6">
        <CardTitle className="text-sm sm:text-base">{label}</CardTitle>
        <CardDescription className="text-xs sm:text-sm">Freshness Classification</CardDescription>
      </CardHeader>
      <CardContent className="p-3 sm:p-6 pt-0 sm:pt-0">
        <div className={`p-3 sm:p-4 rounded-lg border-2 ${colorClass} text-center mb-4`}>
          <p className="font-bold text-base sm:text-lg">{formatClassName(prediction.class)}</p>
          {label !== 'Overall Fish' && (
            <p className="text-xs sm:text-sm mt-1 opacity-90">
              Confidence: {(prediction.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>

        {label === 'Overall Fish' && eyePrediction && gillPrediction ? (
          // Overall Fish with explanation
          <div className="space-y-3">
            <p className="text-xs sm:text-sm">
              <span className="font-semibold">Eye:</span> {eyePrediction.class.replace('_', ' ')} {' '} | {' '}
              <span className="font-semibold">Gill:</span> {gillPrediction.class.replace('_', ' ')}
            </p>
            
            <div className="bg-muted/50 rounded-lg p-3 text-xs sm:text-sm">
              <p className="font-semibold mb-2">How result determined:</p>
              <p className="leading-relaxed text-muted-foreground mb-2">
                System selects the <span className="font-semibold text-foreground">worse (more severe)</span> prediction.
              </p>
              <p className="text-foreground font-medium text-xs">
                {(() => {
                  const severityMap: Record<string, number> = {
                    fresh: 0,
                    less_fresh: 1,
                    starting_to_rot: 2,
                    rotten: 3
                  }
                  const eyeSeverity = severityMap[eyePrediction.class.toLowerCase()] ?? -1
                  const gillSeverity = severityMap[gillPrediction.class.toLowerCase()] ?? -1
                  const selectedSource = eyeSeverity >= gillSeverity ? 'Eye' : 'Gill'
                  const selectedConfidence = selectedSource === 'Eye' ? eyePrediction.confidence : gillPrediction.confidence
                  return `${selectedSource} selected: ${prediction.class.replace('_', ' ')} (${(selectedConfidence * 100).toFixed(1)}%)`
                })()}
              </p>
            </div>
          </div>
        ) : (
          // Eye/Gill with confidence breakdown
          <div className="space-y-2">
            <p className="text-xs sm:text-sm font-semibold text-muted-foreground mb-2">Confidence by Class:</p>
            {['fresh', 'less_fresh', 'starting_to_rot', 'rotten']
              .map((className) => (
                <div key={className} className="flex items-center gap-2">
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs sm:text-sm font-medium">
                        {formatClassName(className)}
                      </span>
                      <span className="text-xs font-semibold text-muted-foreground">
                        {((prediction.probabilities[className] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${getColorForBarClass(className)}`}
                        style={{ width: `${(prediction.probabilities[className] || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function getColorForBarClass(className: string): string {
  const normalized = className.toLowerCase()
  switch (normalized) {
    case 'fresh':
      return 'bg-green-500 dark:bg-green-400'
    case 'less_fresh':
      return 'bg-yellow-500 dark:bg-yellow-400'
    case 'starting_to_rot':
      return 'bg-orange-500 dark:bg-orange-400'
    case 'rotten':
      return 'bg-red-500 dark:bg-red-400'
    default:
      return 'bg-gray-500 dark:bg-gray-400'
  }
}
