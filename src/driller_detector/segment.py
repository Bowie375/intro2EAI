import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

class InteractiveSAM:
    def __init__(self, sam_checkpoint, image_path=None):
        if image_path is not None:
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.display_image = self.image.copy()
        else:
            self.image = None
            self.display_image = None
        
        self.points = []
        self.labels = []
        self.window_name = "Interactive SAM - Left:Foreground, Right:Background, Space:Predict, R:Reset, Q:Quit"
        
        model_type = "vit_h"
        device = "cpu"
        print(f"Loading SAM model on {device}...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        
        # 只有在有图片时才设置图片
        if self.image is not None:
            self.predictor.set_image(self.image)
        
        print("SAM model loaded successfully!")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_point(x, y, 0)
    
    def add_point(self, x, y, label):
        self.points.append([x, y])
        self.labels.append(label)
        
        color = (0, 255, 0) if label == 1 else (0, 0, 255) 
        cv2.circle(self.display_image, (x, y), 8, color, -1)
        cv2.circle(self.display_image, (x, y), 8, (255, 255, 255), 2) 
        
        point_num = len(self.points)
        cv2.putText(self.display_image, f"{point_num}", (x+12, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"Added point {point_num}: ({x}, {y}), label: {'Foreground' if label == 1 else 'Background'}")
        
    def predict_mask(self):
        if len(self.points) == 0:
            print("No points selected! Please click on the image first.")
            return None
        
        print(f"Predicting with {len(self.points)} points...")
        
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            print(f"Generated {len(masks)} masks with scores: {scores}")
            
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            print(f"Selected mask {best_mask_idx} with score: {best_score:.3f}")
            
            mask_overlay = self.image.copy()
            mask_overlay[best_mask] = mask_overlay[best_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
            
            for i, ((x, y), label) in enumerate(zip(self.points, self.labels)):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(mask_overlay, (x, y), 8, color, -1)
                cv2.circle(mask_overlay, (x, y), 8, (255, 255, 255), 2)
                cv2.putText(mask_overlay, f"{i+1}", (x+12, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Mask Result", mask_overlay)
            
            mask_binary = (best_mask * 255).astype(np.uint8)
            # cv2.imwrite("data/interactive_mask.png", mask_binary)
            # cv2.imwrite("data/interactive_result.png", mask_overlay)
            # print("Results saved: 'data/interactive_mask.png' and 'data/interactive_result.png'")
            
            for i, (mask, score) in enumerate(zip(masks, scores)):
                mask_img = (mask * 255).astype(np.uint8)
                # cv2.imwrite(f"data/candidate_mask_{i}_score_{score:.3f}.png", mask_img)
            
            return best_mask
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
    
    def reset_points(self):
        self.points = []
        self.labels = []
        if self.image is not None:
            self.display_image = self.image.copy()
        print("Reset all points.")
    
    def run(self, input_image):
        self.image = input_image
        self.display_image = input_image.copy()
        self.predictor.set_image(input_image)
        self.points = []
        self.labels = []
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== Interactive SAM ===")
        print("Instructions:")
        print("- Left click: Add foreground point (green)")
        print("- Right click: Add background point (red)")
        print("- Press SPACE: Predict mask")
        print("- Press 'r': Reset all points")
        print("- Press 'q': Quit")
        print("========================\n")
        
        seg = None
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord(' '):  # Space key
                seg = self.predict_mask()
            elif key == ord('r'):
                self.reset_points()
        
        cv2.destroyAllWindows()
        return seg

 
 
 
if __name__ == "__main__":
    interactive_sam = InteractiveSAM()
    image = cv2.imread("data/rgb.png")
    seg = interactive_sam.run(image)
    print(seg)
    cv2.imwrite("seg.png", (seg * 255).astype(np.uint8))
    
