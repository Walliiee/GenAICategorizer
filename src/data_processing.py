import os
import json
import glob
import csv
from datetime import datetime
from langdetect import detect
from collections import Counter

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def extract_text_from_conversation(conversation):
    """
    Extracts and concatenates text from conversations in various JSON formats.
    Handles both Claude and ChatGPT conversation structures.
    """
    # Case 1: Claude format with chat_messages array
    if "chat_messages" in conversation:
        texts = []
        for msg in conversation.get("chat_messages", []):
            message_text = ""
            
            # Try direct text field first (common in Claude)
            if "text" in msg and msg["text"]:
                message_text += msg["text"].strip()
                
            # Check content array regardless of whether text field was empty
            content = msg.get("content", [])
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        # Regular text field
                        if "text" in content_item and content_item["text"]:
                            message_text += " " + content_item["text"].strip()
                        # Handle type=text items (common in Claude)
                        elif content_item.get("type") == "text" and "text" in content_item and content_item["text"]:
                            message_text += " " + content_item["text"].strip()
                    elif isinstance(content_item, str) and content_item.strip():
                        message_text += " " + content_item.strip()
            
            # Only add non-empty message text
            if message_text.strip():
                texts.append(message_text.strip())
        
        return " ".join(texts)
    
    # Case 2: ChatGPT format with mapping structure
    elif "mapping" in conversation:
        texts = []
        # Process all messages in the mapping
        for msg_id, msg_data in conversation.get("mapping", {}).items():
            # Skip if no message
            if not msg_data.get("message"):
                continue
                
            message = msg_data["message"]
            content = message.get("content", {})
            
            # Handle different content structures
            if isinstance(content, dict) and "parts" in content:
                parts = content["parts"]
                for part in parts:
                    # Audio transcription type
                    if isinstance(part, dict) and part.get("content_type") == "audio_transcription" and "text" in part:
                        if part["text"].strip():
                            texts.append(part["text"].strip())
                    # Direct text part
                    elif isinstance(part, str) and part.strip():
                        texts.append(part.strip())
        
        return " ".join(texts)
    
    # Fallback: Deep search for text in any JSON structure
    return deep_search_for_text(conversation)

def deep_search_for_text(obj, max_depth=5, current_depth=0):
    """
    Recursively search any JSON object for text fields.
    This is a fallback method for unknown structures.
    """
    if current_depth > max_depth:  # Prevent infinite recursion
        return ""
    
    texts = []
    
    if isinstance(obj, dict):
        # Check common text field names
        for field in ["text", "content", "message"]:
            if field in obj and isinstance(obj[field], str) and obj[field].strip():
                texts.append(obj[field].strip())
        
        # Recursively search all values
        for key, value in obj.items():
            result = deep_search_for_text(value, max_depth, current_depth + 1)
            if result:
                texts.append(result)
    
    elif isinstance(obj, list):
        # Recursively search all items
        for item in obj:
            result = deep_search_for_text(item, max_depth, current_depth + 1)
            if result:
                texts.append(result)
    
    return " ".join(texts)

def process_raw_files(raw_dir, output_csv):
    """
    Reads JSON files, extracts conversation text, and writes non-empty conversations
    to a CSV file, filtering out empty conversations. Also detects language.
    """
    file_paths = glob.glob(os.path.join(raw_dir, "*.json"))
    print(f"Found {len(file_paths)} JSON files in {raw_dir}")
    
    rows = []
    empty_count = 0
    total_conversations = 0
    language_counts = Counter()
    
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Handle file with array of conversations (Claude format)
                if isinstance(data, list):
                    for conv in data:
                        total_conversations += 1
                        conv_id = conv.get("uuid", f"conv_{total_conversations}")
                        conversation_text = extract_text_from_conversation(conv)
                        
                        if conversation_text.strip():
                            # Detect language
                            lang = detect_language(conversation_text)
                            language_counts[lang] += 1
                            
                            # Only add non-empty conversations
                            rows.append({
                                "conversation_id": conv_id,
                                "text": conversation_text,
                                "language": lang
                            })
                        else:
                            empty_count += 1
                
                # Handle single conversation file (either format)
                elif isinstance(data, dict):
                    total_conversations += 1
                    # Try to get ID from either format
                    conv_id = data.get("uuid", data.get("title", f"conv_{total_conversations}"))
                    conversation_text = extract_text_from_conversation(data)
                    
                    if conversation_text.strip():
                        # Detect language
                        lang = detect_language(conversation_text)
                        language_counts[lang] += 1
                        
                        # Only add non-empty conversations
                        rows.append({
                            "conversation_id": conv_id,
                            "text": conversation_text,
                            "language": lang
                        })
                    else:
                        empty_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\nLanguage Statistics:")
    for lang, count in language_counts.most_common():
        print(f"{lang}: {count} conversations")
    
    print(f"\nTotal conversations processed: {total_conversations}")
    print(f"Empty conversations discarded: {empty_count}")
    print(f"Non-empty conversations saved: {len(rows)}")
    print(f"Success rate: {(len(rows) / total_conversations) * 100:.2f}%")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write only non-empty conversations to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["conversation_id", "text", "language"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of script directory)
    project_root = os.path.dirname(script_dir)
    
    # Define paths relative to project root
    raw_dir = os.path.join(project_root, "data", "raw")
    output_csv = os.path.join(project_root, "data", "processed", "cleaned_conversations.csv")
    
    process_raw_files(raw_dir, output_csv)
    