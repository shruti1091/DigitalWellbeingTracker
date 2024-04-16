import streamlit as st
import cv2
import numpy as np
from joblib import load
import pandas as pd
from PIL import Image
from numpy import *
import csv
from joblib import load


def Generate_score(image_path):
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    
    # Use cv2.imdecode() to convert the bytes array into an OpenCV image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (2480, 3508))

    # Save the resized image to your local storage as a PNG file
    cv2.imwrite('image_resized.png', resized_image)

    # Now, `resized_image` holds the resized image array
    image_array = resized_image



    # In[14]:


    # Convert image to grayscale
    gray_scale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    img_bin = cv2.adaptiveThreshold(gray_scale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    # Invert the image if necessary
    img_bin = 255 - img_bin


    # In[15]


    # In[16]:


    # set min width to detect horizontal lines
    line_min_width = 15

    # kernel to detect horizontal lines
    kernal_h = np.ones((1,line_min_width), np.uint8)

    # kernel to detect vertical lines
    kernal_v = np.ones((line_min_width,1), np.uint8)

    # horizontal kernel on the image
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)

    # verical kernel on the image
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)

    # combining the image

    img_bin_final=img_bin_h|img_bin_v

    # In[17]:

    # Detect contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas_list = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Add the area to the list
        if(350<=area<=550):
            areas_list.append(area)

    # Convert the list of areas to a NumPy array
    areas_array = np.array(areas_list)

    # Using mode to find the most common area among the filtered contours
    if len(areas_array) > 0:  # Ensure there are contours after filtering
        # Assuming areas_array is already defined and filled with contour areas
        mean_area = np.mean(areas_array)
        
        # Now, use mean_area as needed
        print("Mean area:", mean_area)
        
        # For setting default values for MIN_CHECKBOX_AREA and MAX_CHECKBOX_AREA based on mean_area
        area_tolerance = 150  # You can adjust this tolerance as needed
        MIN_CHECKBOX_AREA = max(0, mean_area - area_tolerance)
        MAX_CHECKBOX_AREA = mean_area + area_tolerance

    else:
        print("No contours with area > 100 found.")
        # Handle the case where no valid contours are found
        # This might involve setting default values or skipping certain processing steps

    print(MIN_CHECKBOX_AREA, MAX_CHECKBOX_AREA)


    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    checkboxes = []  # List to store information about checkboxes

    # Initialize a counter for the checkbox labels
    checkbox_counter = 1

    for x, y, w, h, area in stats[2:]:
        #print(area)
        if MIN_CHECKBOX_AREA <= area <= MAX_CHECKBOX_AREA:
            # Draw the rectangle on the image
            cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Label the rectangle with a sequential number
            label_position = (x + 5, y + 15)  # Adjust label position as needed
            cv2.putText(image_array, str(checkbox_counter), label_position, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
            checkbox_counter += 1  # Increment the counter
            
            # Save the checkbox details in your list
            checkbox_details = {"x": x, "y": y, "width": w, "height": h, "area": area}
            checkboxes.append(checkbox_details)

    # Display the modified image with labels
    Image.fromarray(image_array).show()


    # In[18]:
    with open('checkboxes.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "width", "height", "area"])  # Header
        for checkbox in checkboxes:
            writer.writerow([checkbox["x"], checkbox["y"], checkbox["width"], checkbox["height"], checkbox["area"]])

    # In[19]:
    # Load the data from CSV
    checkboxes_df = pd.read_csv('checkboxes.csv')

    # Sort the DataFrame by the 'y' coordinate to ensure sequential processing
    checkboxes_df.sort_values(by='y', inplace=True)

    # Initialize variables for processing
    groups = []
    current_group = []
    previous_y = None
    group_number = 0  # Initialize group number

    # Grouping logic based on y-axis proximity and group size, and sorting each group by the x-axis
    for index, row in checkboxes_df.iterrows():
        x, y, width, height, area = row['x'], row['y'], row['width'], row['height'], row['area']
        if previous_y is None or (y - previous_y) <= 10:
            current_group.append(row)
        else:
            # New group due to y-axis gap, check size of the current group
            if len(current_group) >= 25:
                group_number += 1  # Increment group number for each new valid group
                # Sort the current group by the 'x' value
                current_group_sorted = sorted(current_group, key=lambda x: x['x'])
                # Add the group number to each row in the current group
                for item in current_group_sorted:
                    item['group_number'] = group_number
                groups.append(current_group_sorted)
            current_group = [row]  # Start a new group
        previous_y = y

    # Check and sort the last group after exiting the loop
    if len(current_group) >= 25:
        group_number += 1  # Increment group number for the last valid group
        current_group_sorted = sorted(current_group, key=lambda x: x['x'])
        # Add the group number to each row in the current group
        for item in current_group_sorted:
            item['group_number'] = group_number
        groups.append(current_group_sorted)

    # Convert groups back to DataFrame and save to a new CSV file, if not empty
    if groups:
        all_groups_df = pd.concat([pd.DataFrame(group) for group in groups], ignore_index=True)
        all_groups_df.to_csv('filtered_checkboxes_with_groups.csv', index=False)
        print("Filtered checkboxes with group numbers saved to 'filtered_checkboxes_with_groups.csv'.")
    else:
        print("No groups met the criteria.")

    # In[20]:
    # Path to the form image containing all checkboxes
    image_path = 'C:\\Users\\shrih\\anaconda3\\jupyter\\InnovationPractices\\image_resized.png'  

    def extract_resize_prepare(image, x, y, width, height, target_size=(40, 40)):
        # Extract the checkbox based on the provided coordinates
        checkbox_image = image[y:y+height, x:x+width]
        
        # Resize the checkbox to the target size expected by the model
        resized_image = cv2.resize(checkbox_image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to be between 0 and 1
        img_array = np.array(resized_image).flatten() / 255.0
        
        return img_array

    # Load the form image once
    form_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Load the trained Random Forest model
    clf = load('checkbox_classifier.joblib')

    # Load your CSV file with the detected checkboxes
    checkboxes_df = pd.read_csv('filtered_checkboxes_with_groups.csv')

    # Placeholder for classification results
    classification_results = []

    for index, row in checkboxes_df.iterrows():
        img_array = extract_resize_prepare(
            form_image,
            x=row['x'],
            y=row['y'],
            width=row['width'],
            height=row['height']
        )

        # Use the model to predict the classification (marked=1 or unmarked=0)
        prediction = clf.predict([img_array])[0]
        classification_results.append(prediction)

    # Update the DataFrame with the classification results
    checkboxes_df['classification'] = classification_results

    # Save the updated DataFrame back to a CSV file
    checkboxes_df.to_csv('classified_checkboxes.csv', index=False)

    #print("Updated classifications saved to classified_checkboxes.csv")
    score = Calculate_score()
    return score


def Calculate_score():
    screen_time_scores = {
        1: 100,  # 2-3 hours
        2: 75,   # 3-5 hours
        3: 50,   # 5-8 hours
        4: 25    # more than 8 hours
    }

    df = pd.read_csv('classified_checkboxes.csv')

    # Filter the DataFrame for screen time-related entries
    screen_time_df = df[df['group_number'].isin([1, 2, 3, 4])]

    # Initialize an empty list to store scores for all 31 days
    daily_scores = []

    # Assuming entries are sorted chronologically and by group_number
    for i in range(0, len(screen_time_df), 31):  # Step through the dataframe in blocks of 31
        # Extract a block of 31 for the current `group_number`
        block = screen_time_df[i:i+31]
        
        # Assuming 'marked' column exists and indicates if the option was selected
        # We look for the highest group_number marked within each day-block
        marked_blocks = block[block['classification'] == 1]  # Filter out only marked
        if not marked_blocks.empty:
            highest_group_number = marked_blocks['group_number'].max()
            daily_scores.append(screen_time_scores[highest_group_number])
        else:
            # Consider default or minimum score if no option is marked
            daily_scores.append(screen_time_scores[min(screen_time_scores.keys())])

    # Calculate the average score over the period
    average_score = sum(daily_scores) / len(daily_scores)
    #print(f"Average Screen Time Score over the period: {average_score:.2f}")

    # In[10]:

    haleness_scores = {
        5: 33.33,  # Eye break rule
        6: 33.33,  # Exposure to sun
        7: 33.33   # Enable dark mode at night
    }

    # Filter DataFrame for Haleness-related entries
    haleness_df = df[df['group_number'].isin([5, 6, 7])]

    # Initialize a list to store Haleness scores for each day
    haleness_daily_scores = [0] * 31  # Assuming a month of 31 days

    for i in range(0, len(haleness_df), 93):  # Step through the dataframe in blocks of 93 (3 subparts * 31 days)
        for day in range(31):  # For each day
            # Extract entries for the current day within the block
            day_entries = haleness_df.iloc[i + day::31]  # Skip every 31 entries to get the same day across subparts
            
            # Calculate the day's score based on marked entries
            day_score = sum(haleness_scores[gn] for gn, marked in zip(day_entries['group_number'], day_entries['classification']) if marked == 1)
            
            # Store the calculated score
            haleness_daily_scores[day] += day_score

    # Calculate the average Haleness score over the period
    average_haleness_score = sum(haleness_daily_scores) / len(haleness_daily_scores)
    #print(f"Average Haleness Score over the period: {average_haleness_score:.2f}")

    # In[11]:

    water_consumption_scores = {
        8: 25,  # less than 1 litre
        9: 75,  # less than 2 litres
        10: 100  # less than 5 litres
    }

    # Filter DataFrame for Water Consumption-related entries
    water_df = df[df['group_number'].isin([8, 9, 10])]

    # Initialize a list to store Water Consumption scores for each day
    water_daily_scores = [0] * 31  # Assuming a month of 31 days

    for i in range(0, len(water_df), 93):  # Step through the dataframe in blocks of 93 (3 categories * 31 days)
        for day in range(31):  # For each day
            # Extract entries for the current day within the block
            day_entries = water_df.iloc[i + day::32]  # Skip every 31 entries to get the same day across categories
            
            # Calculate the day's score based on marked entries
            # Considering the minimum score among selected options
            day_scores = [water_consumption_scores[gn] for gn, marked in zip(day_entries['group_number'], day_entries['classification']) if marked == 1]
            
            # If multiple selections, consider the minimum score (worst case)
            if day_scores:
                day_min_score = min(day_scores)
            else:
                # Default to 0 if no selection is marked
                day_min_score = 0
            
            # Store the calculated score
            water_daily_scores[day] += day_min_score

    # Calculate the average Water Consumption score over the period
    average_water_score = sum(water_daily_scores) / len(water_daily_scores)
    #print(f"Average Water Consumption Score over the period: {average_water_score:.2f}")

    # In[12]:

    # Since unmarked values contribute positively, and marked values contribute 0,
    # the score for each hassle when it is NOT reported (unmarked) is defined here.
    hassle_scores = {
        11: 20,  # eye strain
        12: 20,  # back pain
        13: 20,  # stressed out
        14: 20,  # battery anxiety
        15: 20   # thumb and wrist discomfort
    }

    # Filter DataFrame for Physical and Mental Hassle-related entries
    hassle_df = df[df['group_number'].isin([11, 12, 13, 14, 15])]

    # Assuming each group_number has 32 relevant entries, with the 32nd to be ignored.
    # Initialize a list to store daily scores
    daily_hassle_scores = [0] * 31  # Assuming a month of 31 days

    for i in range(0, len(hassle_df), 160):  # 5 categories * 32 entries
        for day in range(31):  # For each day
            # Extract entries for the current day across all hassle categories
            day_entries = hassle_df.iloc[i + day::32]  # Skip every 32 entries to align the same day
            
            # Calculate the day's score based on unmarked entries
            day_score = sum(hassle_scores[gn] for gn in day_entries[day_entries['classification'] == 0]['group_number'])
            
            # Store the calculated score
            daily_hassle_scores[day] += day_score

    # Calculate the average score over the period
    average_hassle_score = sum(daily_hassle_scores) / len(daily_hassle_scores)
    print(f"Average Physical and Mental Hassle Score over the period: {average_hassle_score:.2f}")

    # In[13]:

    meal_scores = {
        16: 33.33,  # breakfast
        17: 33.33,  # lunch
        18: 33.33   # dinner
    }

    # Filter DataFrame for Screen Free Meal-related entries
    meal_df = df[df['group_number'].isin([16, 17, 18])]

    # Assuming each group_number has 32 relevant entries, with the 32nd to be ignored.
    # Initialize a list to store daily scores for screen-free meals
    daily_meal_scores = [0] * 31  # Assuming a month of 31 days

    for i in range(0, len(meal_df), 96):  # 3 categories * 32 entries
        for day in range(31):  # For each day
            # Extract entries for the current day across all meal categories
            day_entries = meal_df.iloc[i + day::32]  # Skip every 32 entries to align the same day
            
            # Calculate the day's score based on marked entries (screen-free meals)
            day_score = sum(meal_scores[gn] for gn in day_entries[day_entries['classification'] == 1]['group_number'])
            
            # Store the calculated score
            daily_meal_scores[day] += day_score

    # Calculate the average score over the period for screen-free meals
    average_meal_score = sum(daily_meal_scores) / len(daily_meal_scores)
    #print(f"Average Screen Free Meal Score over the period: {average_meal_score:.2f}")

    # In[14]:

    meal_time_combinations = {
        (19, 20): 100,  # 6-8 am & 6-8 pm
        (19, 22): 85,   # 6-8 am & 8-10 pm
        (19, 24): 70,   # 6-8 am & 10-11 pm
        (21, 20): 85,   # 8-10 am & 6-8 pm
        (21, 22): 70,   # 8-10 am & 8-10 pm
        (21, 24): 55,   # 8-10 am & 10-11 pm
        (23, 20): 70,   # 10-11 am & 6-8 pm
        (23, 22): 55,   # 10-11 am & 8-10 pm
        (23, 24): 40    # 10-11 am & 10-11 pm
    }
    # Assume an additional fallback score for days when only one meal is marked
    fallback_scores = {
        19: 50,  # 6-8 am
        20: 50,  # 6-8 pm
        21: 40,  # 8-10 am
        22: 40,  # 8-10 pm
        23: 30,  # 10-11 am
        24: 30   # 10-11 pm
    }

    # Filter DataFrame for Meal Time-related entries
    meal_time_df = df[df['group_number'].isin([19, 20, 21, 22, 23, 24])]

    # Initialize daily meal time scores, assuming 31 days for simplicity
    daily_meal_time_scores = []

    # Iterate through meal_time_df in blocks representing each day's entries
    for i in range(0, len(meal_time_df), 192):  # Adjust iteration for the actual data structure
        # Identify marked parameters for the day
        marked_params = meal_time_df.iloc[i:i+32][meal_time_df.iloc[i:i+32]['classification'] == 1]['group_number']

        day_score = 0
        # Check for direct combination matches first
        for combo, score in meal_time_combinations.items():
            if set(combo).issubset(marked_params.values):
                day_score = score
                break
        # If no direct match found, apply fallback scoring for single marked meals
        if day_score == 0:
            for param in marked_params:
                day_score += fallback_scores.get(param, 0)
                # Break after first match since we only consider one meal for fallback
                break

        daily_meal_time_scores.append(day_score)

    # Calculate and print the average Meal Time Journal score
    average_meal_time_score = sum(daily_meal_time_scores) / len(daily_meal_time_scores)
    #print(f"Average Meal Time Journal Score over the period: {average_meal_time_score:.2f}")

    # In[15]:

    sleep_quality_scores = {
        25: 100,  # deep sleep - more than 7 hours
        26: 50,   # light sleep - less than 5 hours
        27: 25    # often awake - less than 3 hours
    }

    # Filter DataFrame for Sleep Quality-related entries
    sleep_df = df[df['group_number'].isin([25, 26, 27])]

    
    # Initialize a list to store daily scores for sleep quality
    daily_sleep_scores = [0] * 31  # Assuming a month of 31 days

    for i in range(0, len(sleep_df), 96):  # 3 categories * 32 entries
        for day in range(31):  # For each day
            # Extract entries for the current day across all sleep quality categories
            day_entries = sleep_df.iloc[i + day::32]  # Skip every 32 entries to align the same day
            
            # Calculate the day's score based on the minimum score among marked entries (reflecting the worst condition marked)
            if not day_entries[day_entries['classification'] == 1].empty:
                min_group_number_marked = day_entries[day_entries['classification'] == 1]['group_number'].min()
                day_score = sleep_quality_scores.get(min_group_number_marked, 0)  # Get the score, default to 0 if none marked
                daily_sleep_scores[day] = day_score
            else:
                # Consider the day as deep sleep if no conditions are marked
                daily_sleep_scores[day] = sleep_quality_scores[25]

    # Calculate the average Sleep Quality score over the period
    average_sleep_score = sum(daily_sleep_scores) / len(daily_sleep_scores)
    #print(f"Average Sleep Quality Score over the period: {average_sleep_score:.2f}")

    # In[16]:

    social_interaction_scores = {
        28: 25,  # more on social media
        29: 100  # more in person conversation
    }

    # Filter DataFrame for Social Interaction-related entries
    social_df = df[df['group_number'].isin([28, 29])]

    # Initialize daily social interaction scores, assuming 31 days for simplicity
    daily_social_scores = [0] * 31

    for i in range(0, len(social_df), 64):  # 2 categories * 32 entries (assuming extra entry to be ignored)
        for day in range(31):  # For each day
            # Extract entries for the current day across all social interaction categories
            day_entries = social_df.iloc[i + day::32]  # Skip every 32 entries to align the same day
            
            # Filter 'day_entries' for 'in person conversation' being marked
            in_person_marked = day_entries[(day_entries['group_number'] == 29) & (day_entries['classification'] == 1)]
            
            # Filter 'day_entries' for 'social media' being marked
            social_media_marked = day_entries[(day_entries['group_number'] == 28) & (day_entries['classification'] == 1)]
            
            if not in_person_marked.empty:  # If any 'in person conversation' is marked
                daily_social_scores[day] = social_interaction_scores[29]
            elif not social_media_marked.empty:  # If any 'social media' is marked
                daily_social_scores[day] = social_interaction_scores[28]
            else:
                daily_social_scores[day] = 0  # No social interaction marked

    # Calculate the average Social Interaction score over the period
    average_social_score = sum(daily_social_scores) / len(daily_social_scores)
    #print(f"Average Social Interaction Score over the period: {average_social_score:.2f}")

    # In[17]:

    # Scores out of 100 for each category
    score_screen_time = average_score  # Your calculated score for Screen Time
    score_haleness = average_haleness_score  # Your calculated score for Haleness
    score_water_consumption = average_water_score  # Your calculated score for Water Consumption
    score_physical_mental_hassle = average_hassle_score  # Your calculated score for Physical and Mental Hassle
    score_screen_free_meals = average_meal_score  # Your calculated score for Screen Free Meals
    score_meal_time_journal = average_meal_time_score  # Your calculated score for Meal Time Journal
    score_sleep_quality = average_sleep_score  # Your calculated score for Sleep Quality Indicators
    score_social_interaction = average_social_score  # Your calculated score for Social Interaction

    # Weightage of each category towards the final score
    weight_screen_time = 0.20
    weight_haleness = 0.15
    weight_water_consumption = 0.10
    weight_physical_mental_hassle = 0.15
    weight_screen_free_meals = 0.10
    weight_meal_time_journal = 0.05
    weight_sleep_quality = 0.15
    weight_social_interaction = 0.10

    # Calculate the final score
    final_score = (score_screen_time * weight_screen_time + 
                score_haleness * weight_haleness + 
                score_water_consumption * weight_water_consumption + 
                score_physical_mental_hassle * weight_physical_mental_hassle + 
                score_screen_free_meals * weight_screen_free_meals + 
                score_meal_time_journal * weight_meal_time_journal + 
                score_sleep_quality * weight_sleep_quality + 
                score_social_interaction * weight_social_interaction)

    #print(f"The final digital wellbeing score is: {final_score}")
    return final_score


st.title('Digital Well Being Tracker')
st.write('Please upload your monthly wellness tracker.')

col1, col2 = st.columns(2)

with col1:
    st.header("PDF Upload")
    uploaded_pdf = st.file_uploader("Choose your PDF file", type="pdf", key="pdf_uploader")
    if uploaded_pdf is not None:
        # health_score_pdf = calculate_health_score_from_pdf(uploaded_pdf)
        health_score_pdf = 85  # Placeholder score
        st.success('PDF file successfully uploaded.')
        st.write(f'Your Digital Health Score from PDF is: {health_score_pdf}')

with col2:
    st.header("PNG Upload")
    uploaded_png = st.file_uploader("Choose your PNG file", type="png", key="png_uploader")
    if uploaded_png is not None:
        health_score_png = Generate_score(uploaded_png)
        #health_score_png = 85  # Placeholder score
        st.success('PNG file successfully uploaded.')
        st.write(f'Your Digital Health Score from PNG is: {health_score_png}')
