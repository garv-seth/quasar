"""
Generate PWA icons for QUASAR QA³
This script generates basic icons for the PWA
"""

from PIL import Image, ImageDraw, ImageFont
import os

def generate_icon(size, filename):
    """Generate a simple quantum-themed icon"""
    # Create a new image with a transparent background
    img = Image.new('RGBA', (size, size), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background circle
    circle_diameter = size * 0.9
    circle_offset = (size - circle_diameter) / 2
    draw.ellipse(
        [(circle_offset, circle_offset), 
         (size - circle_offset, size - circle_offset)],
        fill=(123, 47, 255, 255)  # Purple for quantum theme
    )
    
    # Inner circle
    inner_diameter = size * 0.75
    inner_offset = (size - inner_diameter) / 2
    draw.ellipse(
        [(inner_offset, inner_offset), 
         (size - inner_offset, size - inner_offset)],
        fill=(30, 30, 46, 255)  # Dark background
    )
    
    # Draw quantum wave patterns (simplified)
    wave_height = size * 0.05
    wave_width = size * 0.6
    wave_offset_x = (size - wave_width) / 2
    wave_offset_y = size * 0.5
    
    # Draw multiple waves to simulate quantum superposition
    for i in range(3):
        y_offset = wave_offset_y - size * 0.1 + i * size * 0.1
        draw.arc(
            [(wave_offset_x, y_offset - wave_height), 
             (wave_offset_x + wave_width, y_offset + wave_height)],
            start=0, end=180, 
            fill=(123, 47, 255, 200 - i * 40),
            width=int(size * 0.02)
        )
    
    # Draw the Q and 3 symbols
    if size >= 192:  # Only add text on larger icons
        try:
            # Try to load a font, or fall back to default
            font_size = int(size * 0.3)
            try:
                font = ImageFont.truetype("Arial", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            # Draw Q
            draw.text(
                (size * 0.34, size * 0.32),
                "Q",
                fill=(255, 255, 255, 255),
                font=font
            )
            
            # Draw A³
            draw.text(
                (size * 0.48, size * 0.32),
                "A³",
                fill=(123, 47, 255, 255),
                font=font
            )
        except Exception as e:
            print(f"Could not add text to icon: {e}")
    
    # Save the image
    img.save(filename, 'PNG')
    print(f"Generated icon: {filename}")

if __name__ == "__main__":
    # Create icons directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Generate icons
    generate_icon(192, "icon-192.png")
    generate_icon(512, "icon-512.png")
    
    # Generate a screenshot
    screenshot = Image.new('RGB', (1280, 720), color=(30, 30, 46))
    draw = ImageDraw.Draw(screenshot)
    
    # Header
    draw.rectangle([(0, 0), (1280, 60)], fill=(37, 37, 57))
    
    # Sidebar
    draw.rectangle([(0, 0), (300, 720)], fill=(37, 37, 57))
    
    # Main content area background
    draw.rectangle([(320, 80), (1260, 700)], fill=(42, 42, 64))
    
    # Quantum circuit visualization (simplified)
    circuit_height = 200
    circuit_y = 120
    draw.rectangle([(400, circuit_y), (1200, circuit_y + circuit_height)], 
                  fill=(30, 30, 46), outline=(123, 47, 255), width=2)
    
    # Draw qubits and gates
    for i in range(4):
        y = circuit_y + 30 + i * 40
        # Qubit line
        draw.line([(420, y), (1180, y)], fill=(200, 200, 200), width=2)
        
        # Gates
        positions = [500, 600, 700, 900, 1000]
        for pos in positions:
            if (i + pos) % 3 == 0:  # H gates
                draw.rectangle([(pos - 20, y - 20), (pos + 20, y + 20)], 
                              outline=(123, 47, 255), width=2)
                # Add H text
                draw.text((pos - 5, y - 8), "H", fill=(255, 255, 255))
            elif (i + pos) % 3 == 1:  # X gates
                draw.ellipse([(pos - 20, y - 20), (pos + 20, y + 20)], 
                            outline=(123, 47, 255), width=2)
                # Add X text
                draw.text((pos - 5, y - 8), "X", fill=(255, 255, 255))
    
    # Control lines
    draw.line([(700, circuit_y + 30), (700, circuit_y + 150)], 
             fill=(123, 47, 255), width=2)
    draw.line([(900, circuit_y + 70), (900, circuit_y + 190)], 
             fill=(123, 47, 255), width=2)
    
    # Add title
    try:
        # Try to load a font, or fall back to default
        font_size = 36
        try:
            font = ImageFont.truetype("Arial", font_size)
            small_font = ImageFont.truetype("Arial", 24)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        draw.text((320, 20), "QUASAR QA³: Quantum-Accelerated AI Agent", 
                 fill=(255, 255, 255), font=font)
        
        draw.text((320, 350), "Quantum Circuit Visualization", 
                 fill=(255, 255, 255), font=small_font)
        
        draw.text((320, 400), "Results:", 
                 fill=(255, 255, 255), font=small_font)
    except Exception as e:
        print(f"Could not add text to screenshot: {e}")
    
    # Save the screenshot
    screenshot.save("screenshot1.png", 'PNG')
    print("Generated screenshot: screenshot1.png")