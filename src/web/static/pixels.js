/** This file defines functions for handling pixel values */

// Create prototype to avoid redundancy
const canvasPrototype = {
  // Method to draw an image on the HTML canvas
  drawImage(canvas, dx=0, dy=0) {
    this.ctx.drawImage(canvas, dx, dy, this.size, this.size)
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

function getGrayscalePixels(canvas) {
  const smallCanvas = new Canvas()

  smallCanvas.drawImage(canvas, 0, 0)

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