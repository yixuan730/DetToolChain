import ast
from refer.refer import REFER
from tools import *
import base64
from PIL import Image
import re
from io import BytesIO
import backoff
client = OpenAI()


class ObjectLocator:
    def __init__(self):
        self.visual_prompt_toolkits = self.load_visual_prompts()
        self.difficulty_list = self.load_difficulties()
        self.marker_function_map = self.load_marker_function_map()
        self.coor_function_map = self.load_coor_function_map()
        self.zoom_function_map = self.load_zoom_function_map()
        self.reason_function_map = self.load_reason_function_map()

    def load_marker_function_map(self):
        return {
            "compass marker": compass_marker,
            "ruler marker": ruler_marker
        }

    def load_coor_function_map(self):
        return {
            "box proposal marker": box_position_proposal_marker_thought,
            "box marker": box_marker_thought,
            "centroid marker": centroid_marker_thought,
            "number marker": number_marker_thought,
            "convex hull marker": convex_hull_marker_thought,
            'Spatial Relationship Explorer': spatial_relationship_explorer,
        }

    def load_zoom_function_map(self):
        return {
            "zoom in": zoom_in
        }

    def load_reason_function_map(self):
        return {
            "Self-Verification Promoter": self_verification_promoter,
        }

    def load_visual_prompts(self):
        return {
            "Tool: ruler marker": "Helps in adding scale and quadrants marks for normalized coordinates (x1, y1, x2, y2) measurements in images.",
            "Tool: box marker": "Marks the bounding box in image based on given normalized coordinates (x1, y1, x2, y2).",
            "Tool: convex hull marker": "For irregular or obstructed object, predicts the coordinates to form a convex hull and its minimum enclosing rectangle to obtain an approximate box of the object.",
            "Tool: centroid marker": "Marks the center of a given bounding box in image, used to represent the position of a object.",
            "Tool: number marker": "Marks number objects or boxes in image for clear identification.",
            "Tool: box proposal marker": "Based on the given predicted bounding box coordinates and the target object in the image, generates additional box proposals that more accurately fit the target object.",
            "Tool: compass marker": "For rotated objects, Draws reference lines at specific angles for better visual angle judgment in images.",
            "Tool: zoom in": "For small objects or ambiguous objects, magnifies a portion of image to see more detail or to focus on a specific area.",
            "Tool: Self-Verification Promoter": "Check the accuracy or consistency of its own responses, promoting reliability and trustworthiness in its predicted box.",
            "Tool: Spatial Relationship Explorer": "Leverages spatial reasoning ability to analyze the relationships of objects.",
            "Tool: Debate Organizer": "Combines responses from multiple agents' predictions to decide on the best answer.",
        }

    def load_difficulties(self):
        return {
            'small object detection': 'Locates the target object that occupies a small portion of the image. It is challenging due to the limited number of pixels representing the object, which makes it difficult to capture detailed edge.',
            'multi-instance detection': 'The image contains many objects of the same category; the challenge lies in identifying the specific target object referred by the given sentence.',
            'occlusion object detection': 'Deals with detecting target object that are partially obscured by other objects. The key challenge is to correctly identify and locate the whole object despite only a portion of it being visible.'
        }

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_image_from_pil_image(self, img_file):
        buffer = BytesIO()
        img_file.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return encoded_image

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def insights(self, img_file, sent, image_id):
        exp_img_path = './exp/default.jpg'
        exp_text = """For example, in the image, the target object is the old woman, she's in the center of the image.
        To precisely locate the normalized coordinates (x1, y1, x2, y2), call [Tool: ruler marker] to mark scale and quadrant. 
        It is found that her left boundary x1 is near the 0.25 mark on the ruler, 
        her top boundary is near the 0.2 mark on the ruler, 
        her right boundary is near the 0.8 mark on the ruler, 
        and his bottom boundary is on the bottom of the image (i.e., 1). 
        Therefore, her normalized coordinates are (0.25, 0.2, 0.8, 1). """

        test_img = self.encode_image_from_pil_image(img_file)
        response_insights = client.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=[
                {"role": "system",
                 "content": """
                    Your final goal is to locate the target object in the image that described by given sentence, 
                    and then represent the normalized coordinates of the region. 
                    Regions are represented by (x1,y1,x2,y2) coordinates. x1 x2 are the left and right most positions, 
                    normalized into 0 to 1, where 0 is the left and 1 is the right. 
                    y1 y2 are the top and bottom most positions, normalized into 0 to 1, where 0 is the top and 1 is the bottom.
                    Before achieving this final goal, you need to divide the final goal into subtasks and solve them step by step. 
                    In each step, you can call a tool in the pre-defined visual_prompt_toolkits to solve a corresponding subtask. 
                    Each tool has a name and description of its purpose and usage, and the visual_prompt_toolkits is: {}.
                    """.format(self.visual_prompt_toolkits)
                 },
                {"role": "user",
                 "content": [
                     {"type": "text",
                      "text": """Please locate the target object in the image that described by: {}. 
                        To provide you an example, you should determine the type of difficulty from {} and 
                        query corresponding example from [Tool: Problem Insight Guider].
                        Return me one or more difficulties [small object detection, multi-instance detection, occlusion object detection].
                        Note that do not return additional response.
                        """.format(sent, self.difficulty_list)
                      },
                     {"type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{test_img}",
                          "detail": "high"}
                      },
                 ]
                 }
            ],
            temperature=1,
            max_tokens=4096
        )
        insights = response_insights.choices[0].message.content
        print('Insights:', insights)
        if insights == 'small object detection':
            exp_img_path = './exp/small1.jpg'
            exp_text = """For example, in the image, the target object is the small white dog, 
            which is difficult to precisely locate the normalized coordinates (x1, y1, x2, y2). 
            Therefore, call [Tool: ruler marker] to mark scale and quadrant. 
            It is found that the dog is in the fourth quadrant. Then call [Tool: zoom in] to 
            zoom in the fourth quadrant, with a factor of 2.0. The normalized coordinates of the dog 
            in the fourth quadrant are (0.8,0.2,1,0.3). The coordinates are then converted back to the 
            normalized coordinates in the original image is (0.9,0.6,1,0.65).       
            """
        elif insights == 'multi-instance detection':
            exp_img_path = './exp/multi-instance.jpg'
            exp_text = """For example, in the image, there are two instances of elephants. [Tool: number marker] is called 
            to number the elephant on the left as 1, and number the elephant on the right as 2. Then, call [Tool: ruler marker]
            to mark scale and quadrants. The normalized coordinates (x1, y1, x2, y2) for the left one are (0, 0.25, 0.75, 1), 
            and for the right one are (0.5, 0.3, 1, 1). The x1 and x2 values for the left elephant are smaller than those 
            for the right elephant. The trunk of the left elephant extends towards the right side of the image, 
            which is why its x2 coordinate is relatively large, at 0.75. The heights of the two elephants are quite similar, 
            hence their y1 and y2 values are almost the same.
            """
        elif insights == 'occlusion object detection':
            exp_img_path = './exp/occlusion.jpg'
            exp_text = """For example, in this image, the man riding the bike is obscured by a woman standing on the bike. 
            To mark the coordinates of the man's position, it is necessary to locate the coordinates of his various endpoints. 
            Call [Tool: ruler marker] to mark scale and quadrants. His leftmost point is his hand, so x1 is 0.05; 
            his uppermost point is his partially obscured head, so y1 is at 0.3; his rightmost point is the right side of his body, 
            so x2 is 0.4; and his lowest point is his feet, so y2 is 0.8. Thus, the normalized coordinates are (0.05, 0.3, 0.4, 0.8).
            """
        return exp_img_path, exp_text

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def CoT(self, exp_img_path, exp_text, img_file, sent):
        test_img = self.encode_image_from_pil_image(img_file)
        exp_img = self.encode_image(exp_img_path)

        response_cot = client.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=[
                {"role": "system",
                 "content": """
                    Your final goal is to locate the target object in the image that described by given sentence, 
                    and then represent the normalized coordinates of the region. 
                    Before achieving this final goal, you need to divide the final goal into subtasks and solve them step by step. 
                    In each step, you can call a tool in the pre-defined visual_prompt_toolkits {} to solve a corresponding subtask. 
                    There are some key points to consider:
                    (1) If there are multiple objects in the image, the Spatial Relationship Explorer can be used to analyze the relationships of objects.
                    (2) It is advisable to have self-verification at the end to confirm the consistency of historical prediction records.
                    (3) zoom in is very effective for small objects.
                    (4) If you think the target in the current image is difficult to predict, 
                    you can use the Debate Organizer to combine responses from multiple agents' predictions.
                    """.format(self.visual_prompt_toolkits)
                 },
                {"role": "assistant",
                 "content": [
                     {"type": "text",
                      "text": """{}""".format(exp_text)},
                     {"type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{exp_img}",
                          "detail": "high"}
                      },
                 ]
                 },
                {"role": "user",
                 "content": [
                     {"type": "text",
                      "text": """The final goal is to locate the target object in the BELOW image that described by: {}. 
                      Following the example, how can we achieve the final goal step by step? 
                      Return in the following format and replace the content with needed tool name in the visual_prompt_toolkits:
                      Tool: tool_name; Tool: tool_name; Tool: tool_name; Tool: tool_name; ...
                      Do not return additional response.
                    """.format(sent)
                      },
                     {"type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{test_img}",
                          "detail": "high"}
                      },
                 ]
                 }
            ],
            temperature=1,
            max_tokens=4096
        )
        cot_str = response_cot.choices[0].message.content
        print('CoT:', cot_str)
        tools_list = re.findall(r'Tool:\s*([^;]+)', cot_str)
        print(tools_list)
        return tools_list

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def final(self, img_file, sent, coor, tool_list):
        test_img = self.encode_image_from_pil_image(img_file)
        response = client.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=[
                {"role": "system",
                 "content": """
            Your final goal is to locate the target object in the image that described by given sentence, 
            and then represent the normalized coordinates of the region. 
            Regions are represented by (x1,y1,x2,y2) coordinates. x1 x2 are the left and right most positions, 
            normalized into 0 to 1, where 0 is the left and 1 is the right. 
            y1 y2 are the top and bottom most positions, normalized into 0 to 1, where 0 is the top and 1 is the bottom.
            Before achieving this final goal, you need to divide the final goal into subtasks and solve them step by step. 
            In each step, you can call a tool in the pre-defined visual_prompt_toolkits to solve a corresponding subtask. 
            Each tool has a name and description of its purpose and usage, and the visual_prompt_toolkits is: {}.
            """.format(self.visual_prompt_toolkits)
                 },
                {"role": "user",
                 "content": [
                     {"type": "text",
                      "text": """Locate the target object in the BELOW image that described by: {}. 
                         The tools you have used are: {}. 
                         The normalized coordinates of the target object are: {}.
                         Based on the information given, what is the most logical next Tool should be called? return: Tool: tool_name 
                         Or if you verify that this is the final prediction of coordinates, return: I've got the final answer 
                         Do not return any other information.
                            """.format(sent, tool_list, coor)
                      },
                     {"type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{test_img}",
                          "detail": "high"}
                      },
                 ]
                 }
            ],
            temperature=1,
            max_tokens=4096
        )
        final = response.choices[0].message.content
        print('The Next Step:', final)
        if 'final' in final:
            return 'I\'ve got the final answer'
        else:
            tools_list = re.findall(r'Tool:\s*([^;]+)', final)
            return tools_list

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def locate(self, img_file, sent, exp_img_path=None, exp_text=None):
        test_img = self.encode_image_from_pil_image(img_file)
        if exp_text is None:
            print('without exp')
            response = client.chat.completions.create(
                # model='gpt-4-vision-preview',
                model='gpt-4o-2024-05-13',
                messages=[
                    {"role": "system",
                     "content": """Analyze the spatial relationships of objects in the image and determine the region of the target object.
                Regions are represented by (x1,y1,x2,y2) coordinates. x1 x2 are the left and right most positions,
                normalized into 0 to 1, where 0 is the left and 1 is the right.
                y1 y2 are the top and bottom most positions, normalized into 0 to 1, where 0 is the top and 1 is the bottom."""
                     },
                    {"role": "user",
                     "content": [
                         {"type": "text",
                          "text": """Locate the target object in the BELOW image that described by: {}.
                                      Return the normalized coordinates in the following format and do not return other information:
                                      (x1,y1,x2,y2)cc c
                                    """.format(sent)
                          },
                         {"type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{test_img}",
                              "detail": "high"}
                          },
                     ]
                     }
                ],
                temperature=1,
                max_tokens=4096
            )
        else:
            print('with exp')
            exp_img = self.encode_image(exp_img_path)
            response = client.chat.completions.create(
                model='gpt-4-vision-preview',
                messages=[
                    {"role": "system",
                     "content": """
                Your final goal is to locate the target object in the image that described by given sentence, 
                and then represent the normalized coordinates of the region. 
                Regions are represented by (x1,y1,x2,y2) coordinates. x1 x2 are the left and right most positions, 
                normalized into 0 to 1, where 0 is the left and 1 is the right. 
                y1 y2 are the top and bottom most positions, normalized into 0 to 1, where 0 is the top and 1 is the bottom.
                """
                     },
                    {"role": "assistant",
                     "content": [
                         {"type": "text",
                          "text": """{}""".format(exp_text)},
                         {"type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{exp_img}",
                              "detail": "high"}
                          },
                     ]
                     },
                    {"role": "user",
                     "content": [
                         {"type": "text",
                          "text": """Locate the target object in the BELOW image that described by: {}. 
                          Return the normalized coordinates in the following format and do not return other information:
                          (x1,y1,x2,y2)
                        """.format(sent)
                          },
                         {"type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{test_img}",
                              "detail": "high"}
                          },
                     ]
                     }
                ],
                temperature=1,
                max_tokens=4096
            )
        coor = response.choices[0].message.content
        print('normalized coordinates:', coor)
        return ast.literal_eval(coor)

    def execute_tools(self, tool_list, img, sent, coor):
        origin_img = img
        current_img = img
        origin_coor = coor
        current_coor = coor
        processed_img_dict = {}
        inter_coor_dict = {}
        for i, tool_name in enumerate(tool_list):
            if tool_name in self.marker_function_map:
                func = self.marker_function_map[tool_name]
                current_img = func(current_img, sent, current_coor)
                processed_img_dict[f'step_{i + 1}_img'] = current_img
                inter_coor_dict[f'step_{i + 1}_img'] = current_coor
            elif tool_name in self.coor_function_map:
                func = self.coor_function_map[tool_name]
                current_coor = func(current_img, sent, current_coor)
                processed_img_dict[f'step_{i + 1}_img'] = current_img
                inter_coor_dict[f'step_{i + 1}_img'] = current_coor
            elif tool_name in self.zoom_function_map:
                func = self.zoom_function_map[tool_name]
                last_img = current_img
                current_img, split_flag = func(current_img, sent, current_coor)
                if split_flag:
                    print("The image has been split and quadrant{} has been zoomed.".format(split_flag))
                    current_coor = self.locate(current_img, sent)
                    current_img = last_img
                    current_coor = quadrant2whole(split_flag, current_coor)
                processed_img_dict[f'step_{i + 1}_img'] = current_img
                inter_coor_dict[f'step_{i + 1}_img'] = current_coor
            elif tool_name in self.reason_function_map:
                func = self.reason_function_map[tool_name]
                current_coor = func(origin_img, sent, current_coor, inter_coor_dict)
                processed_img_dict[f'step_{i + 1}_img'] = origin_img
                inter_coor_dict[f'step_{i + 1}_img'] = current_coor
            else:
                print("Tool not found.")
        return current_img, current_coor, processed_img_dict, inter_coor_dict


if __name__ == "__main__":
    data_root = './refer/data'  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = 'refcoco'
    splitBy = 'unc'
    refer = REFER(data_root, dataset, splitBy)
    ref_ids = refer.getRefIds()
    ref_ids = refer.getRefIds(split='testA')
    print('There are %s training referred objects.' % len(ref_ids))
    fail_ref_ids = []
    max_attempts = 3
    for ref_id in ref_ids[9:10]:
        attempts = 0
        success = False
        while attempts < max_attempts and not success:
            try:
                # ref_id = 1407
                ref = refer.loadRefs(ref_id)[0]
                # print(ref)
                img_path = './refer/data/images/mscoco/images/train2014/COCO_train2014_' + \
                           ref['file_name'].split('_')[2] + '.jpg'
                img = Image.open(img_path)
                width, height = img.size
                sent = ref['sentences'][0]['sent']
                sents_list = [sentence['sent'] for sentence in ref['sentences']]
                print(sents_list)
                image_id = ref['image_id']
                ann_id = ref['ann_id']
                bbox_ab = refer.getRefBox(ref['ref_id'])
                bbox_normal = (
                    bbox_ab[0] / width, bbox_ab[1] / height, (bbox_ab[0] + bbox_ab[2]) / width,
                    (bbox_ab[1] + bbox_ab[3]) / height)
                locator = ObjectLocator()
                exp_img_path, exp_text = locator.insights(img, sent, image_id)
                tools_list = locator.CoT(exp_img_path, exp_text, img, sent)
                if 'Debate Organizer' in tools_list:
                    current_img_debate = {}
                    current_coor_debate = {}
                    processed_img_dict_debate = {}
                    inter_coor_dict_debate = {}
                    for i, sent in enumerate(sents_list):
                        print('Debating sentence:', sent)
                        coor0 = locator.locate(img, sent, exp_img_path=None, exp_text=None)
                        current_img, current_coor, processed_img_dict, inter_coor_dict = locator.execute_tools(
                            tools_list, img, sent, coor0)
                        current_img_debate[i] = current_img
                        current_coor_debate[i] = current_coor
                        processed_img_dict_debate[i] = processed_img_dict
                        inter_coor_dict_debate[i] = inter_coor_dict
                    best_i = debate(img, sents_list, current_img_debate, current_coor_debate, processed_img_dict_debate,
                                    inter_coor_dict_debate)
                    current_img, current_coor, processed_img_dict, inter_coor_dict = current_img_debate[best_i], \
                                                                                     current_coor_debate[best_i], \
                                                                                     processed_img_dict_debate[best_i], \
                                                                                     inter_coor_dict_debate[best_i]
                else:
                    coor0 = locator.locate(img, sent, exp_img_path=None, exp_text=None)
                    current_img, current_coor, processed_img_dict, inter_coor_dict = locator.execute_tools(tools_list,
                                                                                                           img, sent,
                                                                                                           coor0)

                print(inter_coor_dict)
                final_flag = locator.final(img, sent, current_coor, tools_list)
                while not isinstance(final_flag, str) and attempts < max_attempts:
                    current_img, current_coor, processed_img_dict, inter_coor_dict = locator.execute_tools(final_flag,
                                                                                                           img, sent,
                                                                                                           current_coor)
                    tools_list.extend(final_flag)
                    final_flag = locator.final(img, sent, current_coor, tools_list)
                    attempts += 1
                    print(f"Attempt {attempts} of {max_attempts} for more tools.")

                if isinstance(final_flag, str) or attempts >= max_attempts:
                    print('Finished!')
                    print(current_coor)
                    success = True

            except Exception as e:
                print(f"Error processing ref_id {ref_id}: {e}")
                fail_ref_ids.append(ref_id)
                attempts += 1
                print(f"Attempt {attempts} of {max_attempts} for more attempts.")
                print(f"Failed ref_ids: {fail_ref_ids}")
