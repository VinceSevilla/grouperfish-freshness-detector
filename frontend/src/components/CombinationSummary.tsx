import { Prediction } from '@/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface CombinationSummaryProps {
  eyePrediction: Prediction | null
  gillPrediction: Prediction | null
}

function getCombinationSummary(eyeClass: string, gillClass: string): string {
  const eye = eyeClass.toLowerCase()
  const gill = gillClass.toLowerCase()

  // All 16 possible combinations
  if (eye === 'fresh' && gill === 'fresh') {
    return 'Both eyes and gills in excellent condition - ideal for consumption'
  }
  if (eye === 'fresh' && gill === 'less_fresh') {
    return 'Eyes are fresh but gills show slight aging - acceptable but use soon'
  }
  if (eye === 'fresh' && gill === 'starting_to_rot') {
    return 'Eyes are fresh but gills are deteriorating - caution recommended'
  }
  if (eye === 'fresh' && gill === 'rotten') {
    return 'Eyes are fresh but gills are rotten - not suitable for consumption'
  }

  if (eye === 'less_fresh' && gill === 'fresh') {
    return 'Gills are fresh but eyes show slight aging - acceptable but use soon'
  }
  if (eye === 'less_fresh' && gill === 'less_fresh') {
    return 'Both eyes and gills show signs of aging - acceptable if used promptly'
  }
  if (eye === 'less_fresh' && gill === 'starting_to_rot') {
    return 'Eyes are aging and gills are deteriorating - use with care'
  }
  if (eye === 'less_fresh' && gill === 'rotten') {
    return 'Eyes are aging and gills are rotten - not recommended'
  }

  if (eye === 'starting_to_rot' && gill === 'fresh') {
    return 'Gills are good but eyes are deteriorating - suspicious quality'
  }
  if (eye === 'starting_to_rot' && gill === 'less_fresh') {
    return 'Eyes are deteriorating and gills are aging - handle with care'
  }
  if (eye === 'starting_to_rot' && gill === 'starting_to_rot') {
    return 'Both eyes and gills are deteriorating - not recommended for consumption'
  }
  if (eye === 'starting_to_rot' && gill === 'rotten') {
    return 'Eyes are deteriorating and gills are rotten - do not consume'
  }

  if (eye === 'rotten' && gill === 'fresh') {
    return 'Eyes are rotten but gills appear fresh - questionable quality'
  }
  if (eye === 'rotten' && gill === 'less_fresh') {
    return 'Eyes are rotten and gills are aging - not trustworthy'
  }
  if (eye === 'rotten' && gill === 'starting_to_rot') {
    return 'Eyes are rotten and gills are deteriorating - do not consume'
  }
  if (eye === 'rotten' && gill === 'rotten') {
    return 'Both eyes and gills are rotten - definitely not suitable for consumption'
  }

  return 'Unable to determine freshness'
}

function getRecommendationColor(eyeClass: string, gillClass: string): string {
  const eye = eyeClass.toLowerCase()
  const gill = gillClass.toLowerCase()

  // Safe combinations (green)
  if ((eye === 'fresh' && (gill === 'fresh' || gill === 'less_fresh')) ||
      (eye === 'less_fresh' && (gill === 'fresh' || gill === 'less_fresh'))) {
    return 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700'
  }

  // Caution combinations (orange)
  if ((eye === 'fresh' && gill === 'starting_to_rot') ||
      (eye === 'less_fresh' && (gill === 'starting_to_rot')) ||
      (eye === 'starting_to_rot' && (gill === 'fresh' || gill === 'less_fresh'))) {
    return 'bg-orange-50 dark:bg-orange-900/20 border-orange-300 dark:border-orange-700'
  }

  // Unsafe combinations (red)
  return 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700'
}

export function CombinationSummary({ eyePrediction, gillPrediction }: CombinationSummaryProps) {
  if (!eyePrediction || !gillPrediction) {
    return null
  }

  const summary = getCombinationSummary(eyePrediction.class, gillPrediction.class)
  const colorClass = getRecommendationColor(eyePrediction.class, gillPrediction.class)

  return (
    <Card className="md:col-span-2 lg:col-span-3">
      <CardHeader>
        <CardTitle className="text-base">Combined Assessment</CardTitle>
        <CardDescription>Summary based on both eye and gill analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <div className={`p-4 rounded-lg border-2 ${colorClass}`}>
          <p className="text-sm leading-relaxed">{summary}</p>
        </div>
      </CardContent>
    </Card>
  )
}
