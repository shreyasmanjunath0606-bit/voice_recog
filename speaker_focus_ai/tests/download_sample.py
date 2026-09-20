import urllib.request
import os

def download_sample_video():
    # URL of a sample video (Big Buck Bunny is a common public domain test video)
    # Ideally, for speaker focus, we would use a dataset like AVA or LRS3, 
    # but for testing pipeline structure, this works.
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "sample.mp4")
    
    if os.path.exists(output_path):
        print(f"Sample video already exists at {output_path}")
        return
        
    print(f"Downloading sample video to {output_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download complete.")

if __name__ == "__main__":
    download_sample_video()
