import { useRef, useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Modal } from '@/components/ui/modal'
import { PredictionDisplay } from '@/components/PredictionDisplay'
import { CombinationSummary } from '@/components/CombinationSummary'
import { PredictionWarningModal } from '@/components/PredictionWarningModal'
import { predictionsAPI } from '@/utils/api'
import { PredictionResponse } from '@/types'

export function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null)
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [showResultsModal, setShowResultsModal] = useState(false)
  const [showWarningModal, setShowWarningModal] = useState(false)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment")
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [showFlash, setShowFlash] = useState(false)

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  const startCamera = async () => {
    try {
      setCameraError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
      }
    } catch (error) {
      setCameraError('Unable to access camera. Please check permissions.')
      console.error('Error accessing camera:', error)
    }
  }

  const toggleCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
    if (isStreaming) {
      stopCamera();
      setTimeout(() => startCamera(), 300);
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      setIsStreaming(false)
      setCapturedImage(null)
    }
  }

  const captureFrame = (): string | null => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext('2d')
      if (context) {
        // Capture exactly what's displayed on screen
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)
        return canvas.toDataURL('image/jpeg', 0.9)
      }
    }
    return null
  }

  const analyzeFrame = async () => {
    try {
      setIsPredicting(true)
      
      // Show flash effect
      setShowFlash(true)
      setTimeout(() => setShowFlash(false), 200)
      
      const base64Image = captureFrame()

      if (!base64Image) {
        setCameraError('Failed to capture frame')
        setIsPredicting(false)
        return
      }

      // Show captured image while analyzing
      setCapturedImage(base64Image)

      // Remove data URL prefix
      const imageData = base64Image.replace(/^data:image\/\w+;base64,/, '')

      const response = await predictionsAPI.cameraPredict(imageData)
      setPredictions(response)
      if (response.annotated_image) {
        setAnnotatedImage(`data:image/png;base64,${response.annotated_image}`)
      }
      
      // Check if there are missing predictions
      const eyeMissing = !response.eye_prediction || !response.eye_detected
      const gillMissing = !response.gill_prediction || !response.gill_detected
      
      if (eyeMissing || gillMissing) {
        setShowWarningModal(true)
      } else {
        setShowResultsModal(true)
      }
    } catch (error) {
      setCameraError('Error analyzing frame. Please try again.')
      console.error('Error predicting:', error)
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <div className="container max-w-screen-2xl py-8">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Live Camera Detection</h1>
          <p className="text-muted-foreground mt-2">
            Point your camera at the fish to detect freshness in real-time
          </p>
        </div>

        {/* Camera Section - Centered */}
        <div className="flex justify-center">
          <div className="w-full max-w-2xl">
            <Card>
              <CardHeader>
                <CardTitle>Camera Feed</CardTitle>
                <CardDescription>Point at the fish for detection</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  {!isStreaming && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
                      <div className="text-center space-y-4">
                        <p className="text-white text-lg">Camera not active</p>
                        <Button onClick={startCamera} className="w-full">
                          Start Camera
                        </Button>
                      </div>
                    </div>
                  )}
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className={`w-full h-full object-cover ${isPredicting ? 'hidden' : ''}`}
                  />
                  
                  {/* Show captured image while analyzing */}
                  {isPredicting && capturedImage && (
                    <img 
                      src={capturedImage} 
                      alt="Captured frame" 
                      className="w-full h-full object-cover"
                    />
                  )}

                  {/* Flash effect overlay */}
                  {showFlash && (
                    <div className="absolute inset-0 bg-white animate-pulse z-30" />
                  )}

                  {/* Loading indicator */}
                  {isPredicting && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/40 z-20">
                      <div className="text-center space-y-3">
                        <div className="inline-block">
                          <div className="animate-spin rounded-full h-12 w-12 border-4 border-white border-t-blue-500" />
                        </div>
                        <p className="text-white text-sm font-medium">Analyzing Image...</p>
                      </div>
                    </div>
                  )}
                  
                  {/* Switch Camera Icon Button */}
                  {isStreaming && !isPredicting && (
                    <button
                      onClick={toggleCamera}
                      className="absolute bottom-3 right-3 bg-white/80 hover:bg-white rounded-full p-2 shadow-lg z-20"
                      aria-label="Switch Camera"
                      type="button"
                    >
                      {/* Inline SVG for camera switch icon */}
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-700">
                        <path d="M7 7h.01" />
                        <path d="M21 13v-2a4 4 0 0 0-4-4H7.5l1.5-2" />
                        <path d="M3 11v2a4 4 0 0 0 4 4h9.5l-1.5 2" />
                        <polyline points="7 7 3 11 7 15" />
                        <polyline points="17 7 21 11 17 15" />
                      </svg>
                    </button>
                  )}
                </div>

                {cameraError && (
                  <div className="p-3 bg-destructive/10 border border-destructive rounded-lg text-sm text-destructive">
                    {cameraError}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3">
                  <Button
                    onClick={isStreaming ? stopCamera : startCamera}
                    variant={isStreaming ? 'outline' : 'default'}
                    className="w-full"
                  >
                    {isStreaming ? 'Stop Camera' : 'Start Camera'}
                  </Button>
                  <Button
                    onClick={analyzeFrame}
                    disabled={!isStreaming || isPredicting}
                    className="w-full"
                  >
                    {isPredicting ? 'Analyzing...' : 'Analyze'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      <Modal
        isOpen={showResultsModal}
        onClose={() => {
          setShowResultsModal(false)
          setCapturedImage(null)
        }}
        title="Fish Freshness Analysis Results"
        size="xl"
      >
        <div className="space-y-6">
          {/* Annotated Image */}
          {annotatedImage && (
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-4">Detected Regions</h3>
              <div className="bg-muted rounded-lg p-4">
                <img
                  src={annotatedImage}
                  alt="Annotated Detection"
                  className="max-w-lg h-auto mx-auto rounded shadow-md"
                />
              </div>
            </div>
          )}

          {/* Analysis Results */}
          {predictions && (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Eye Analysis */}
              {predictions.eye_detected && predictions.eye_prediction ? (
                <PredictionDisplay
                  prediction={predictions.eye_prediction}
                  label="Eye"
                  detected={predictions.eye_detected}
                />
              ) : (
                <PredictionDisplay
                  prediction={null}
                  label="Eye"
                  detected={false}
                />
              )}

              {/* Gill Analysis */}
              {predictions.gill_detected && predictions.gill_prediction ? (
                <PredictionDisplay
                  prediction={predictions.gill_prediction}
                  label="Gill"
                  detected={predictions.gill_detected}
                />
              ) : (
                <PredictionDisplay
                  prediction={null}
                  label="Gill"
                  detected={false}
                />
              )}

              {/* Overall Analysis */}
              {predictions.integrated_prediction && (
                <PredictionDisplay
                  prediction={predictions.integrated_prediction}
                  label="Overall Fish"
                  detected={true}
                />
              )}

              {/* Combination Summary */}
              {predictions.eye_prediction && predictions.gill_prediction && (
                <CombinationSummary
                  eyePrediction={predictions.eye_prediction}
                  gillPrediction={predictions.gill_prediction}
                />
              )}
            </div>
          )}

          {/* Warning for no detection */}
          {predictions && !predictions.eye_detected && !predictions.gill_detected && (
            <Card className="bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800">
              <CardContent className="pt-6">
                <p className="text-yellow-800 dark:text-yellow-200 text-center">
                  No eyes or gills detected in the image. Please ensure the fish with visible eyes or gills is clearly shown.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Modal Actions */}
          <div className="flex justify-center space-x-4 pt-4 border-t">
            <Button onClick={() => setShowResultsModal(false)} variant="outline">
              Close Results
            </Button>
            <Button onClick={() => setPredictions(null)}>
              Analyze New Frame
            </Button>
          </div>
        </div>
      </Modal>

      {/* Prediction Warning Modal */}
      {predictions && (
        <PredictionWarningModal
          isOpen={showWarningModal}
          onClose={() => {
            setShowWarningModal(false)
            setCapturedImage(null)
          }}
          onRetry={() => {
            setShowWarningModal(false)
            setShowResultsModal(true)
          }}
          missingPredictions={{
            eyeMissing: !predictions.eye_prediction || !predictions.eye_detected,
            gillMissing: !predictions.gill_prediction || !predictions.gill_detected
          }}
        />
      )}

      {/* Hidden canvas for capturing frames */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
