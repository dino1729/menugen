import os
import sys
import logging
from typing import Dict
from openai import AsyncOpenAI

# Set up logging
logger = logging.getLogger("menugen.openai_description")

# Read OpenAI API key and initialize client
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

# Initialize the async OpenAI client
client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# Read OpenAI description model from environment variable
DESCRIPTION_MODEL = os.getenv("DESCRIPTION_MODEL", "model-router")

async def simplify_menu_item_description_openai(item: Dict) -> str:
    """Takes a menu item dictionary and returns a simplified description using OpenAI.
    If no description exists, it generates one based on the item name.
    Ensures the description is a full sentence and removes surrounding quotes.
    
    Args:
        item: Dictionary containing menu item with 'name' and optional 'description'
        
    Returns:
        Simplified or generated description string
        
    Raises:
        SystemExit: If critical error occurs (terminates application)
    """
    item_name = item.get('name', 'Unknown')
    description = item.get('description')
    logger.info(f"simplify_menu_item_description_openai called for item: {item_name}")
    
    try:
        if description:
            logger.info(f"Simplifying existing description for: {item_name}")
            prompt = (
                f"Rephrase the following menu item description as a single, complete sentence in simple English. "
                f"Explain any potentially unfamiliar culinary terms (like picadillo, aioli, etc.) or the dish name itself (like Pastelón) in simple terms within the sentence. "
                f"The goal is for someone completely unfamiliar with the dish or terms to understand what it is. "
                f"Focus on key ingredients and preparation. Avoid jargon. Do not include any quotation marks in the final output. "
                f"Example 1: If the original description is 'baked Roman-style, semolina gnocchi; gorgonzola cheese, rosemary; salsa rossa', "
                f"the rephrased sentence should be like 'These Roman-style baked dumplings made from semolina flour are served with gorgonzola cheese, rosemary, and a vibrant red sauce.'. "
                f"Example 2: If the item name is 'Pastelón' and description is 'beef picadillo, sweet plantain, cheese fondue', "
                f"the rephrased sentence should be like 'Pastelón is a layered casserole, similar to lasagna, made with seasoned ground beef (picadillo), sweet plantains, and melted cheese.'. "
                f"Original item name: '{item_name}'. Original description: '{description}'"
                f"Rephrased sentence in simple English:"
            )
            system_message = "You rephrase menu descriptions into single, simple English sentences, explaining unfamiliar terms clearly and avoiding quotes."
        else:
            logger.info(f"Generating description for item with no description: {item_name}")
            prompt = (
                f"Generate a simple, concise, and appetizing description for the menu item named '{item_name}' as a single, complete sentence in simple English. "
                f"If the item name itself might be unfamiliar (like 'Pastelón'), briefly explain what it is. "
                f"Focus on likely key ingredients and preparation method based on the name. Avoid jargon. "
                f"Do not include any quotation marks in the final output. "
                f"Example: For an item named 'Focaccia', the generated sentence could be 'Enjoy our freshly baked Italian flatbread, known as Focaccia, perfect for starting your meal.'. "
                f"Generate a description for '{item_name}'."
                f"Generated sentence in simple English:"
            )
            system_message = "You generate simple and appetizing menu descriptions as single, complete sentences in simple English, explaining unfamiliar item names clearly and avoiding quotes."
        
        logger.info(f"Calling OpenAI API for description processing with model: {DESCRIPTION_MODEL}")
        
        # OpenAI parameters
        response = await client.chat.completions.create(
            model=DESCRIPTION_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.6
        )
        
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logger.warning(f"OpenAI API returned no description/simplification for: {item_name}. Returning original or empty.")
            # Clean quotes even from fallback
            clean_description = description.strip().strip('\"') if description else ""
            return clean_description
        
        generated_description = response.choices[0].message.content.strip()
        # Explicitly remove leading/trailing quotes
        generated_description = generated_description.strip('\"')
        logger.info(f"Successfully generated/simplified description for: {item_name}")
        return generated_description
    
    except Exception as e:
        logger.error(f"Error processing description for item {item_name}: {e}")
        logger.critical(f"Terminating application due to critical error in simplify_menu_item_description_openai for item {item_name}.")
        sys.exit(1)  # Terminate the application on error

