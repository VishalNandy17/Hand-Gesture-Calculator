import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# Initializing pipeline
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initializing variables for calculation
expression = []
gesture_history = deque(maxlen=10)
last_gesture = None

def get_finger_state(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_state = [0, 0, 0, 0, 0]

    for i, tip in enumerate(finger_tips):
        if i == 0:  # Thumb
            if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x:
                finger_state[i] = 1
        else:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                finger_state[i] = 1

    return finger_state

def get_finger_angles(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    finger_bases = [1, 5, 9, 13, 17]
    finger_tips = [4, 8, 12, 16, 20]
    angles = []

    for base, tip in zip(finger_bases, finger_tips):
        base_point = hand_landmarks.landmark[base]
        tip_point = hand_landmarks.landmark[tip]
        
        angle = math.degrees(math.atan2(tip_point.y - wrist.y, tip_point.x - wrist.x) - 
                             math.atan2(base_point.y - wrist.y, base_point.x - wrist.x))
        angle = (angle + 360) % 360
        angles.append(angle)

    return angles

def interpret_gesture(finger_state, finger_angles):
    # Basic gestures (0-9)
    if finger_state == [0, 0, 0, 0, 0]:  # Fist
        return 0
    elif finger_state == [0, 1, 0, 0, 0]:  # Index finger
        return 1
    elif finger_state == [0, 1, 1, 0, 0]:  # Peace sign
        return 2
    elif finger_state == [0, 1, 1, 1, 0]:  # Three fingers
        return 3
    elif finger_state == [0, 1, 1, 1, 1]:  # Four fingers
        return 4
    elif finger_state == [1, 1, 1, 1, 1]:  # Open hand
        return 5
    elif finger_state == [1, 1, 1, 0, 0] and finger_angles[0] < 45:  # Thumb and first two fingers, thumb close to index
        return 6
    elif finger_state == [1, 1, 1, 1, 0] and finger_angles[0] < 45:  # Thumb and first three fingers, thumb close to index
        return 7
    elif finger_state == [1, 0, 0, 0, 1] and finger_angles[0] > 90:  # Thumb and pinky spread
        return 8
    elif finger_state == [1, 1, 0, 0, 1] and finger_angles[0] > 90:  # Thumb, index, and pinky spread
        return 9
    
    # Operator gestures
    elif finger_state == [1, 0, 0, 0, 0]:  # Thumb only
        return '+'
    elif finger_state == [0, 0, 0, 0, 1]:  # Pinky only
        return '-'
    elif finger_state == [1, 1, 0, 0, 1]:  # Thumb, index, pinky
        return '*'
    elif finger_state == [1, 0, 0, 0, 1] and finger_angles[0] < 90:  # Thumb and pinky close
        return '/'
    elif finger_state == [0, 1, 0, 1, 0]:  # Index and ring
        return '='
    elif finger_state == [0, 1, 0, 1, 1]:  # Index, ring, pinky
        return 'C'
    elif finger_state == [1, 1, 1, 0, 1]:  # Thumb, index, middle, pinky
        return '^'  # Power operation
    elif finger_state == [1, 0, 1, 0, 1]:  # Thumb, middle, pinky
        return '%'  # Modulo operation
    elif finger_state == [0, 1, 1, 1, 0]:  # Index, middle, ring
        return '.'  # Decimal point
    elif finger_state == [1, 1, 0, 1, 0]:  # Thumb, index, ring
        return '('  # Open parenthesis
    elif finger_state == [1, 1, 0, 1, 1]:  # Thumb, index, ring, pinky
        return ')'  # Close parenthesis
    elif finger_state == [1, 0, 1, 1, 0]:  # Thumb, middle, ring
        return '√'  # Square root
    else:
        return None

def get_stable_gesture(gesture):
    gesture_history.append(gesture)
    if len(gesture_history) == gesture_history.maxlen:
        if all(g == gesture_history[0] for g in gesture_history):
            return gesture_history[0]
    return None

def draw_gesture(image, gesture):
    h, w, _ = image.shape
    gesture_area = np.zeros((200, 200, 3), dtype=np.uint8)
    
    if isinstance(gesture, int):
        cv2.putText(gesture_area, str(gesture), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
    elif gesture in ['+', '-', '*', '/', '^', '%', '.', '(', ')', '√']:
        cv2.putText(gesture_area, gesture, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
    elif gesture == 'C':
        cv2.putText(gesture_area, "CLR", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    elif gesture == '=':
        cv2.putText(gesture_area, "=", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
    
    image[h-220:h-20, w-220:w-20] = gesture_area
    return image

def evaluate_expression(expr):
    try:
        expr = expr.replace('√', 'math.sqrt')
        # eval() to calculate the result
        return eval(expr, {"__builtins__": None}, {"math": math})
    except:
        return "Error"

def draw_result(image, text, position, font_scale=1, thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_offset_x, text_offset_y = position
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def draw_interface(image):
    # title
    cv2.putText(image, "Hand Gesture Calculator", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # gesture guide
    instructions = [
        "0-9: Show fingers",
        "+: Thumb, -: Pinky, *: Thumb+Index+Pinky",
        "/: Thumb+Pinky close, ^: Thumb+Index+Middle+Pinky",
        "%: Thumb+Middle+Pinky, .: Index+Middle+Ring",
        "=: Index+Ring, C: Index+Ring+Pinky",
        "(: Thumb+Index+Ring, ): Thumb+Index+Ring+Pinky",
        "√: Thumb+Middle+Ring"
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(image, instruction, (10, image.shape[0] - 20 - 20*i), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # interface
    image = draw_interface(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_state = get_finger_state(hand_landmarks)
            finger_angles = get_finger_angles(hand_landmarks)
            gesture = interpret_gesture(finger_state, finger_angles)
            stable_gesture = get_stable_gesture(gesture)
            
            if stable_gesture is not None and stable_gesture != last_gesture:
                last_gesture = stable_gesture
                if stable_gesture == 'C':
                    expression = []
                elif stable_gesture == '=':
                    result = evaluate_expression(''.join(map(str, expression)))
                    expression = [str(result)]
                else:
                    expression.append(stable_gesture)
                
            if gesture is not None:
                image = draw_gesture(image, gesture)

    # Displaying expression and result
    display_text = ''.join(map(str, expression))
    draw_result(image, f"Expression: {display_text}", (10, 70), 1, 2)
    
    if len(expression) > 0 and expression[-1] == '=':
        result = evaluate_expression(''.join(map(str, expression[:-1])))
        result_text = f"Result: {result}"
        draw_result(image, result_text, (10, 120), 1.5, 3, (0, 255, 255), (0, 0, 128))

    cv2.imshow('Hand Gesture Calculator', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
