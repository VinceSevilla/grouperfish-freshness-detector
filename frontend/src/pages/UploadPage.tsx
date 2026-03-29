import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Modal } from '@/components/ui/modal'
import { PredictionDisplay } from '@/components/PredictionDisplay'
import { CombinationSummary } from '@/components/CombinationSummary'
import { PredictionWarningModal } from '@/components/PredictionWarningModal'
import { predictionsAPI } from '@/utils/api'
import { PredictionResponse } from '@/types'

export function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [rotation, setRotation] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [predictions, setPredictions] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showResultsModal, setShowResultsModal] = useState(false)
  const [showWarningModal, setShowWarningModal] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragOverRef = useRef(false)
  const previewImageRef = useRef<HTMLImageElement>(null)

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    }

    setSelectedFile(file)
    setRotation(0)
    setError(null)
    setPredictions(null)
    setShowResultsModal(false)

    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const rotateImage = (degrees: number) => {
    setRotation((prev) => (prev + degrees + 360) % 360)
  }

  const getRotatedImageFile = async (): Promise<File> => {
    if (rotation === 0 || !previewImageRef.current || !selectedFile) {
      return selectedFile!
    }

    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!
        
        // Calculate new dimensions based on rotation
        const radians = (rotation * Math.PI) / 180
        const cos = Math.abs(Math.cos(radians))
        const sin = Math.abs(Math.sin(radians))
        
        canvas.width = img.width * cos + img.height * sin
        canvas.height = img.height * cos + img.width * sin
        
        // Move to center, rotate, and draw
        ctx.translate(canvas.width / 2, canvas.height / 2)
        ctx.rotate(radians)
        ctx.drawImage(img, -img.width / 2, -img.height / 2)
        
        canvas.toBlob((blob) => {
          const rotatedFile = new File([blob!], selectedFile!.name, {
            type: selectedFile!.type,
            lastModified: Date.now(),
          })
          resolve(rotatedFile)
        }, selectedFile!.type)
      }
      img.src = preview!
    })
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    dragOverRef.current = true
  }

  const handleDragLeave = () => {
    dragOverRef.current = false
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    dragOverRef.current = false
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const analyzeImage = async () => {
    if (!selectedFile) return

    try {
      setIsLoading(true)
      setError(null)
      
      // Get rotated file if rotation applied, otherwise use original
      const fileToAnalyze = await getRotatedImageFile()
      
      const response = await predictionsAPI.uploadAndPredict(fileToAnalyze)
      setPredictions(response)
      
      // Check if there are missing predictions
      const eyeMissing = !response.eye_prediction || !response.eye_detected
      const gillMissing = !response.gill_prediction || !response.gill_detected
      
      if (eyeMissing || gillMissing) {
        setShowWarningModal(true)
      } else {
        setShowResultsModal(true)
      }
    } catch (err: any) {
      setError('Error analyzing image. Please try again.')
      console.error('Error predicting:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const resetForm = () => {
    setSelectedFile(null)
    setPreview(null)
    setRotation(0)
    setPredictions(null)
    setError(null)
    setShowResultsModal(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="container max-w-screen-2xl py-8">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold">Upload Fish Image</h1>
          <p className="text-muted-foreground mt-2">
            Upload an image of a fish to detect freshness. Drag and drop or click to select.
          </p>
        </div>

        {/* Main Content - Centered */}
        <div className="flex justify-center">
          <div className={`transition-all duration-500 ${selectedFile ? 'w-full max-w-2xl' : 'w-full max-w-xl'}`}>
            <Card>
              <CardHeader className="text-center">
                <CardTitle>Select Image</CardTitle>
                <CardDescription>Drag and drop your fish image here</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Drag and Drop Area - Collapses after upload */}
                {!selectedFile && (
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all duration-300 ${
                      dragOverRef.current
                        ? 'border-primary bg-primary/10'
                        : 'border-muted-foreground/25 hover:border-primary/50'
                    }`}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileInputChange}
                      className="hidden"
                    />

                    <div className="space-y-4">
                      <div className="flex justify-center">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-black dark:text-white">
                          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                        </svg>
                      </div>
                      <div>
                        <p className="text-lg font-semibold">Click to upload or drag and drop</p>
                        <p className="text-muted-foreground">PNG, JPG, JPEG, GIF up to 10MB</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Uploaded Image Preview - Centered */}
                {preview && (
                  <div className="space-y-4">
                    <div className="flex justify-center">
                      <div className="relative bg-muted rounded-lg overflow-visible max-w-md">
                        <img 
                          ref={previewImageRef}
                          src={preview} 
                          alt="Preview" 
                          className="w-full h-auto transition-transform duration-200"
                          style={{
                            transform: `rotate(${rotation}deg)`,
                          }}
                        />
                        
                        {/* Rotation Controls - Bottom Right Corner */}
                        <div className="absolute bottom-2 right-2 flex gap-2">
                          <Button
                            onClick={() => rotateImage(-90)}
                            variant="outline"
                            size="sm"
                            title="Rotate Left"
                            className="h-10 w-10 p-0 bg-white/90 hover:bg-white dark:bg-slate-900/90 dark:hover:bg-slate-900 dark:text-white border-gray-300 dark:border-gray-600 backdrop-blur-sm"
                          >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M1 4v6h6"/>
                              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                            </svg>
                          </Button>
                          <Button
                            onClick={() => rotateImage(90)}
                            variant="outline"
                            size="sm"
                            title="Rotate Right"
                            className="h-10 w-10 p-0 bg-white/90 hover:bg-white dark:bg-slate-900/90 dark:hover:bg-slate-900 dark:text-white border-gray-300 dark:border-gray-600 backdrop-blur-sm"
                          >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M23 4v6h-6"/>
                              <path d="M20.49 15a9 9 0 1 1-2.13-9.36L23 10"/>
                            </svg>
                          </Button>
                        </div>
                      </div>
                    </div>

                    <p className="text-center text-sm font-medium">
                      {selectedFile?.name}
                    </p>
                  </div>
                )}

                {error && (
                  <div className="p-4 bg-destructive/10 border border-destructive rounded-lg text-sm text-destructive text-center">
                    {error}
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex justify-center space-x-4">
                  <Button
                    onClick={resetForm}
                    variant="outline"
                    className={`${selectedFile ? "" : "hidden"} dark:bg-slate-900 dark:hover:bg-slate-800 dark:text-white dark:border-slate-700`}
                  >
                    Upload New Image
                  </Button>
                  <Button
                    onClick={analyzeImage}
                    disabled={!selectedFile || isLoading}
                    variant="outline"
                    className="dark:bg-slate-900 dark:hover:bg-slate-800 dark:text-white dark:border-slate-700"
                  >
                    {isLoading ? 'Analyzing...' : 'Analyze Image'}
                  </Button>
                </div>

                {predictions && (
                  <div className="text-center space-y-3">
                    <Button 
                      onClick={() => setShowResultsModal(true)}
                      variant="outline"
                      className="dark:bg-slate-900 dark:hover:bg-slate-800 dark:text-white dark:border-slate-700"
                    >
                      View Result
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Results Modal */}
      <Modal
        isOpen={showResultsModal}
        onClose={() => setShowResultsModal(false)}
        title="Fish Freshness Analysis Results"
        size="xl"
      >
        <div className="space-y-6">
          {/* Annotated Image */}
          {predictions?.annotated_image && (
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-4">Detected Regions</h3>
              <div className="bg-muted rounded-lg p-4">
                <img
                  src={`data:image/png;base64,${predictions.annotated_image}`}
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
            <Button 
              onClick={resetForm} 
              variant="outline"
              className="dark:bg-slate-900 dark:hover:bg-slate-800 dark:text-white dark:border-slate-700"
            >
              Upload New Image
            </Button>
            <Button 
              onClick={() => setShowResultsModal(false)}
              variant="outline"
              className="dark:bg-slate-900 dark:hover:bg-slate-800 dark:text-white dark:border-slate-700"
            >
              Close Results
            </Button>
          </div>
        </div>
      </Modal>

      {/* Prediction Warning Modal */}
      {predictions && (
        <PredictionWarningModal
          isOpen={showWarningModal}
          onClose={() => setShowWarningModal(false)}
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
    </div>
  )
}