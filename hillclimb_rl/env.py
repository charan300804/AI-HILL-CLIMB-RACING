# env.py

import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time

class HillClimbEnv:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.last_car_x = None
        self.restart_coords = (300, 300)  # Update this to your restart button position
        self.last_action = None
        self.gas_hold_frames = 0
        self.last_state = None

    def _get_screenshot(self):
        img = pyautogui.screenshot(region=(0, 0, 800, 480))  # Adjust to your game window
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def _parse_detections(self, results):
        state = np.zeros(5)
        car_x = None

        for box in results[0].boxes:
            cls = int(box.cls)
            x_center = float(box.xywh[0][0])
            if cls == 1:  # car
                car_x = x_center
                state[0] = x_center
            elif cls == 0: state[1] = 1  # fuel_level
            elif cls == 4: state[2] = 1  # fuel_can
            elif cls == 2: state[3] = 1  # coin
            elif cls == 3: state[4] = 1  # diamond

        return state, car_x

    def _calculate_reward(self, state, car_x):
        # If we can't calculate distance, set reward to 0 or -10 if car is gone
        if car_x is None or self.last_car_x is None:
            self.last_car_x = car_x
            return -10 if car_x is None else 0

        # Base reward on distance moved forward
        reward = car_x - self.last_car_x
        self.last_car_x = car_x

        # Reward shaping
        if state[2]: reward += 10  # fuel_can
        if state[3]: reward += 3   # coin
        if state[4]: reward += 5   # diamond
        if reward < 0.05: reward -= 2  # punish small movement

        # Encourage gas
        if self.last_action == 1:
            reward += 0.5

        return reward
    



    
    def __init__(self, model_path=None):
        self.model = YOLO(model_path)
        self.last_car_x = None
        self.last_state = None
        self.last_action_time = time.time()

    def _press_key(self, action):
        # 0: GAS | 1: BRAKE | 2: NO ACTION

        # Release both keys to avoid conflict
        pyautogui.keyUp('right')
        pyautogui.keyUp('left')

        if action == 1:
            # 🟢 GAS: Hold for 2 seconds to move forward smoothly
            print("🟢 Gas pressed smoothly")
            pyautogui.keyDown('right')
            time.sleep(0.2)
            pyautogui.keyUp('right')
            pyautogui.keyDown('left')
            time.sleep(0.1)
            pyautogui.keyUp('left')
            pyautogui.keyDown('right')
            time.sleep(0.5)
            pyautogui.keyUp('right')
            pyautogui.keyDown('right')
            time.sleep(1.0)
            pyautogui.keyUp('right')
            pyautogui.keyDown('left')
            time.sleep(0.4)
            pyautogui.keyUp('left')
            pyautogui.keyDown('right')
            time.sleep(2.0)
            pyautogui.keyUp('right')

        elif action == 0:
            # 🔴 BRAKE: Only a short tap unless flying
        
                print("🔴 Brake tapped")
                pyautogui.keyDown('left')
                time.sleep(0.3)
                pyautogui.keyUp('left')
           

        else:
            print("⚪ No key action")

        self.last_action_time = time.time()


    def _auto_restart_game(self):
        pyautogui.keyUp("right")
        pyautogui.keyUp("left")
        pyautogui.keyDown('right')
        time.sleep(2)

    def reset(self):
        self.last_car_x = None
        self.last_action = None
        self.gas_hold_frames = 0
        self._auto_restart_game()
        time.sleep(1)
        return np.zeros(5)

    def step(self, action):
        self._press_key(action)
        time.sleep(0.1)
        frame = self._get_screenshot()
        results = self.model(frame)
        state, car_x = self._parse_detections(results)
        reward = self._calculate_reward(state, car_x)
        done = reward == -10

        if done:
            pyautogui.keyUp("right")
            pyautogui.keyUp("left")
            self.last_action = None

        self.last_state = state
        return state, reward, done, results
