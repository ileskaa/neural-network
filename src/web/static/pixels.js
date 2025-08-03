/** This file defines functions for handling pixel values */

// Create prototype to avoid redundancy
const canvasPrototype = {
  // Method to draw an image on the HTML canvas
  drawImage(canvas, dx = 0, dy = 0, size = this.size) {
    this.ctx.drawImage(canvas, dx, dy, size, size)
  },

  // Return object representing the underlying pixel data of the canvas
  getImageData() {
    return this.ctx.getImageData(
      0, 0, this.size, this.size
    )
  }
}

// Constructor function
function Canvas(size = 28) {
  this.size = size
  this.element = document.createElement('canvas')
  this.element.height = this.element.width = size
  this.ctx = this.element.getContext('2d', { alpha: false })

  // By default the canvas has no background. It is transparent.
  // But MNIST consists of white strokes on black backgrounds.
  // We therefore set a white background, so that we can later invert it to black.
  this.ctx.fillStyle = 'white'
  this.ctx.fillRect(0, 0, size, size)
  // After setting the background, every pixel has RGB values of 255,
  // since every pixel is white.
  // Each pixel alpha is also 255, since we don't have transparent pixels anymore.

  // All the pixels in the new ImageData object are transparent black by default
  this.imageData = this.ctx.createImageData(size, size)
  // RGBA pixel values
  this.data = this.imageData.data
}

// Assign methods to the constructor function's prototype
Object.assign(Canvas.prototype, canvasPrototype)

// Display pre-processed digit
function displayPixels(pixels) {
  const canvas = new Canvas()

  // Fill the canvas' imageData object with pixel values
  for (let i = 0; i < pixels.length; i++) {
    const pixelValue = pixels[i]
    // Multiply by 4 since every pixel has 4 values (RGBA)
    const startIndex = i * 4

    // Since we are working in grayscale, we can set R,G and B to the same value
    canvas.data[startIndex] = pixelValue     // R
    canvas.data[startIndex + 1] = pixelValue // G
    canvas.data[startIndex + 2] = pixelValue // B
    canvas.data[startIndex + 3] = 255        // A (no transparency)
  }

  // Paint data from the ImageData object onto the canvas
  canvas.ctx.putImageData(canvas.imageData, 0, 0)

  const div = document.getElementById('pre-processed')
  // Clear container to avoid stacking canvases
  div.innerHTML = ''
  div.appendChild(canvas.element)
}

const sum = (values) => values.reduce(
  (accumulator, pixelVal) => accumulator + pixelVal,
  initialVal
)

function calculateCenter(pixels) {
  initialVal = 0
  const sumPixels = sum(pixels)

  const length = 28

  xCoordinates = yCoordinates = Array.from({ length }, (_, i) => i)

  const xProducts = []
  const yProducts = []

  // For each pixel, we compute two products:
  // one with the x-coordinate and one with the y-coordinate
  for (let i = 0; i < pixels.length; i++) {
    const x = i % length
    const y = Math.floor(i / length)

    const pixel = pixels[i]

    xProducts.push(pixel * x)
    yProducts.push(pixel * y)
  }

  // Compute center for x-coordinates
  const xCenter = sum(xProducts) / sumPixels

  // Compute center for y-coordinates
  const yCenter = sum(yProducts) / sumPixels

  return { xCenter, yCenter }
}

function getInvertedPixels(data) {
  const pixels = []
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]

    // According to W3C, this formula can be used to determine color brightness
    const brightness = r * 0.299 + g * 0.587 + b * 0.114

    // Turn the background to black and the strokes to white
    const inverted = 255 - brightness

    pixels.push(inverted)
  }
  return pixels
}

function getShiftedPixels(canvas, shiftX, shiftY) {
  const canvasShifted = new Canvas()

  canvasShifted.drawImage(canvas, shiftX, shiftY)

  const imageData = canvasShifted.getImageData()
  const data = imageData.data

  const pixels = getInvertedPixels(data)

  const { xCenter, yCenter } = calculateCenter(pixels)

  console.log('xCenter, yCenter', xCenter, yCenter)

  return pixels
}

function extractPixelData(canvasElement) {
  const ctx = canvasElement.getContext('2d')
  const size = canvasElement.height
  const imageData = ctx.getImageData(0, 0, size, size)
  return imageData.data
}

function getBoundingBox(pixels) {
  // There are 280*280 pixels and 4 values for each pixel (RGBA) 
  const size = Math.sqrt(pixels.length / 4)
  let minX = size - 1
  let minY = size - 1
  let maxX = 0
  let maxY = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = (y * size + x) * 4
      const alpha = pixels[index + 3]
      // If the pixel is non-transparent, something has been drawn on it
      if (alpha > 0) {
        minX = Math.min(minX, x)
        minY = Math.min(minY, y)
        maxX = Math.max(maxX, x)
        maxY = Math.max(maxY, y)
      }
    }
  }
  const width = maxX - minX
  const height = maxY - minY
  const boundedRectSize = Math.max(height, width)
  console.log('boundedRectSize:', boundedRectSize)
  return { minX, minY, width, height }
}

function makeClippedCanvas(canvas, minX, minY, width, height) {
  const size = Math.max(width, height)
  const clippedCanvasElt = document.createElement('canvas')
  clippedCanvasElt.width = clippedCanvasElt.height = size
  const clippedCtx = clippedCanvasElt.getContext('2d')

  const offsetX = (size - width) / 2
  const offsetY = (size - height) / 2

  clippedCtx.drawImage(
    canvas, minX, minY, width, height, offsetX, offsetY, width, height
  )
  return clippedCanvasElt
}

function resizeCanvas(canvasElt, size) {
  const smallCanvasElt = document.createElement('canvas')
  smallCanvasElt.width = smallCanvasElt.height = size
  const smallCtx = smallCanvasElt.getContext('2d')
  smallCtx.drawImage(canvasElt, 0, 0, size, size)
  return smallCanvasElt
}

function getGrayscalePixels(canvas) {
  const pixelData = extractPixelData(canvas)
  const { minX, minY, width, height } = getBoundingBox(pixelData)
  const clippedCanvasElt = makeClippedCanvas(canvas, minX, minY, width, height)
  const smallCanvasElt = resizeCanvas(clippedCanvasElt, 20)

  // Create the 28x28 px canvas. We'll put the 20x20 px canvas inside it.
  // This matches the MNIST dataset's processing
  const smallCanvas = new Canvas()

  const offset = 4
  const boundedSize = 20
  smallCanvas.drawImage(canvas, offset, offset, boundedSize, boundedSize)

  const size = smallCanvas.size

  const smallImageData = smallCanvas.getImageData()

  // The data property represents a 1D array containg pixel values in RGBA order.
  // The order goes by rows from the top-left pixel to the bottom-right
  const data = smallImageData.data

  const pixels = getInvertedPixels(data)

  let { xCenter, yCenter } = calculateCenter(pixels)

  xCenter = Math.round(xCenter)
  yCenter = Math.round(yCenter)
  console.log('xCenter, yCenter', xCenter, yCenter)

  const shiftX = size / 2 - xCenter
  const shiftY = size / 2 - yCenter
  const shiftedPixels = getShiftedPixels(smallCanvas.element, shiftX, shiftY)

  displayPixels(shiftedPixels)

  return shiftedPixels
}