# upgradation date 07.01.2024 - added hold gesture with fist and holding same expression for 5s 
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import time

# Setting up MediaPipe - found these confidence values work best after testing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# These help me keep track of the gestures and numbers
expression = []  # Stores the full math expression
gesture_history = deque(maxlen=10)  # Helps smooth out gesture recognition
last_gesture = None
current_number = ""
last_number_time = 0  # Added this to handle holding numbers
last_number_gesture = None  # Keeps track of repeated numbers

def get_finger_state(hand_landmarks):
    """
    Checks which fingers are up or down. Returns a list of 1s and 0s.
    I'm using landmark points from MediaPipe's hand model:
    - Fingers are numbered from thumb (0) to pinky (4)
    - 1 means finger is up, 0 means down
    """
    finger_tips = [4, 8, 12, 16, 20]  # MediaPipe's landmark points for fingertips
    finger_state = [0, 0, 0, 0, 0]

    for i, tip in enumerate(finger_tips):
        # Thumb is special - check if it's pointing left
        if i == 0:
            if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x:
                finger_state[i] = 1
        # Other fingers - check if they're pointing up
        else:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                finger_state[i] = 1

    return finger_state

def get_finger_angles(hand_landmarks):
    """
    This was tricky to figure out! It measures angles between fingers and wrist.
    Really helps distinguish between similar gestures.
    """
    wrist = hand_landmarks.landmark[0]
    finger_bases = [1, 5, 9, 13, 17]  # Where fingers connect to palm
    finger_tips = [4, 8, 12, 16, 20]  # Fingertips
    angles = []

    for base, tip in zip(finger_bases, finger_tips):
        base_point = hand_landmarks.landmark[base]
        tip_point = hand_landmarks.landmark[tip]
        
        # Using atan2 for angle calculation - math class finally useful!
        angle = math.degrees(math.atan2(tip_point.y - wrist.y, tip_point.x - wrist.x) - 
                           math.atan2(base_point.y - wrist.y, base_point.x - wrist.x))
        angle = (angle + 360) % 360
        angles.append(angle)

    return angles

def interpret_gesture(finger_state, finger_angles):
    """
    This is where the magic happens! Turns finger positions into numbers and operators.
    I spent a lot of time fine-tuning these gestures to feel natural.
    
    The gestures I chose:
    - Numbers 0-5: Pretty intuitive finger counting
    - Numbers 6-9: Some creative combinations I came up with
    - Operators: Tried to make them memorable (like thumb up for plus)
    """
    # Basic number gestures
    if finger_state == [0, 0, 0, 0, 0]:  # Fist = 0
        return 0
    elif finger_state == [0, 1, 0, 0, 0]:  # Index = 1
        return 1
    elif finger_state == [0, 1, 1, 0, 0]:  # Peace = 2
        return 2
    elif finger_state == [0, 1, 1, 1, 0]:  # Three fingers
        return 3
    elif finger_state == [0, 1, 1, 1, 1]:  # Four fingers
        return 4
    elif finger_state == [1, 1, 1, 1, 1]:  # All fingers = 5
        return 5
    elif finger_state == [1, 1, 1, 0, 0] and finger_angles[0] < 45:  # 6
        return 6
    elif finger_state == [1, 1, 1, 1, 0] and finger_angles[0] < 45:  # 7
        return 7
    elif finger_state == [1, 0, 0, 0, 1] and finger_angles[0] > 90:  # 8
        return 8
    elif finger_state == [1, 1, 0, 0, 1] and finger_angles[0] > 90:  # 9
        return 9
    
    # My operator gestures - tried to make them intuitive
    elif finger_state == [1, 0, 0, 0, 0]:  # Thumb only = add
        return '+'
    elif finger_state == [0, 0, 0, 0, 1]:  # Pinky only = subtract
        return '-'
    elif finger_state == [1, 1, 0, 0, 1]:  # Thumb, index, pinky = multiply
        return '*'
    elif finger_state == [1, 0, 0, 0, 1] and finger_angles[0] < 90:  # Close thumb-pinky = divide
        return '/'
    elif finger_state == [0, 1, 0, 1, 0]:  # Index and ring = equals
        return '='
    elif finger_state == [0, 1, 0, 1, 1]:  # Index, ring, pinky = clear
        return 'C'
    elif finger_state == [1, 1, 1, 0, 1]:  # Four fingers special = power
        return '^'
    elif finger_state == [1, 0, 1, 0, 1]:  # Thumb, middle, pinky = modulo
        return '%'
    elif finger_state == [0, 1, 1, 1, 0]:  # Three middle fingers = decimal
        return '.'
    elif finger_state == [1, 1, 0, 1, 0]:  # Thumb, index, ring = left parenthesis
        return '('
    elif finger_state == [1, 1, 0, 1, 1]:  # Thumb, index, ring, pinky = right parenthesis
        return ')'
    elif finger_state == [1, 0, 1, 1, 0]:  # Thumb, middle, ring = square root
        return '√'
    else:
        return None

def get_stable_gesture(gesture):
    """
    Makes sure a gesture is held steady before accepting it.
    Helps avoid accidental inputs while moving hands around.
    """
    gesture_history.append(gesture)
    if len(gesture_history) == gesture_history.maxlen:
        if all(g == gesture_history[0] for g in gesture_history):
            return gesture_history[0]
    return None

def draw_gesture(image, gesture):
    """
    Shows the recognized gesture in the corner of the screen.
    Really helpful for debugging and user feedback!
    """
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
    """
    Evaluates the math expression safely.
    Added square root support because why not!
    """
    try:
        expr = expr.replace('√', 'math.sqrt')
        return eval(expr, {"__builtins__": None}, {"math": math})
    except:
        return "Error"

def draw_result(image, text, position, font_scale=1, thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    """
    Draws text with a nice background box.
    Makes it much easier to read the expression and result!
    """
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_offset_x, text_offset_y = position
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def draw_interface(image):
    """
    Adds title and instructions to the video feed.
    I like keeping the instructions visible - helps when learning the gestures!
    """
    # Title at the top
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
        "√: Thumb+Middle+Ring",
        "Hold number gesture for 5s to repeat"
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(image, instruction, (10, image.shape[0] - 20 - 20*i), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

# Main loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip image so it feels more natural (like a mirror)
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Add my interface elements
    image = draw_interface(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks - helps with positioning
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the gesture info
            finger_state = get_finger_state(hand_landmarks)
            finger_angles = get_finger_angles(hand_landmarks)
            gesture = interpret_gesture(finger_state, finger_angles)
            stable_gesture = get_stable_gesture(gesture)
            
            current_time = time.time()
            
            # Handle the recognized gesture
            if stable_gesture is not None:
                if isinstance(stable_gesture, int):  # Number gesture
                    if stable_gesture == last_number_gesture:
                        # Cool feature: hold a number to repeat it!
                        if current_time - last_number_time >= 5:
                            current_number += str(stable_gesture)
                            last_number_time = current_time
                    else:
                        # New number
                        last_number_gesture = stable_gesture
                        last_number_time = current_time
                        if stable_gesture != last_gesture:
                            current_number += str(stable_gesture)
                else:  # Operator gesture
                    last_number_gesture = None
                    if stable_gesture != last_gesture:
                        last_gesture = stable_gesture
                        if stable_gesture == 'C':  # Clear everything
                            expression = []
                            current_number = ""
                        elif stable_gesture == '=':  # Calculate result
                            if current_number:
                                expression.append(current_number)
                                current_number = ""
                            result = evaluate_expression(''.join(map(str, expression)))
                            expression = [str(result)]
                        else:  # Add operator to expression
                            if current_number:
                                expression.append(current_number)
                                current_number = ""
                            expression.append(stable_gesture)
                
            # Show the current gesture
            if gesture is not None:
                image = draw_gesture(image, gesture)

    # Show the expression and result
    display_text = ''.join(map(str, expression)) + current_number
    draw_result(image, f"Expression: {display_text}", (10, 70), 1, 2)
    
    # Show the calculated result
    if len(expression) > 0 and expression[-1] == '=':
        result = evaluate_expression(''.join(map(str, expression[:-1])))
        result_text = f"Result: {result}"
        draw_result(image, result_text, (10, 120), 1.5, 3, (0, 255, 255), (0, 0, 128))

    # Display everything
    cv2.imshow('Hand Gesture Calculator', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
