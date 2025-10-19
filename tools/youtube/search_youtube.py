async def search_youtube(query: str):
    """Search youtube for a specific channel/episode etc"""
    logger.info(f"Searching YouTube for: {query}")
    
    try:
        # Import necessary modules
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # Configure Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Initialize the driver
        driver = webdriver.Chrome(options=options)
        
        try:
            # Construct search URL
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            logger.info(f"Fetching search URL: {search_url}")
            
            # Load the page
            driver.get(search_url)
            
            # Wait for video results to load
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#contents ytd-video-renderer")))
            
            # Extract video information
            videos = []
            video_elements = driver.find_elements(By.CSS_SELECTOR, "div#contents ytd-video-renderer")
            
            for element in video_elements[:10]:  # Limit to first 10 results
                try:
                    # Extract video details
                    title_element = element.find_element(By.CSS_SELECTOR, "#video-title")
                    title = title_element.text.strip()
                    video_url = title_element.get_attribute("href")
                    
                    # Extract channel name
                    try:
                        # Try multiple selectors to find the channel name
                        selectors = [
                            "ytd-channel-name #text-container yt-formatted-string a",
                            "ytd-channel-name #text a",
                            "#channel-info ytd-channel-name a",
                            "#channel-name a"
                        ]
                        channel = ""
                        channel_url = ""
                        for selector in selectors:
                            try:
                                channel_element = element.find_element(By.CSS_SELECTOR, selector)
                                channel = channel_element.text.strip()
                                channel_url = channel_element.get_attribute("href")
                                if channel:  # If we found a non-empty channel name, break
                                    break
                            except:
                                continue
                        
                        if not channel:  # If no channel name found, try one last method
                            channel_element = element.find_element(By.CSS_SELECTOR, "ytd-channel-name")
                            channel = channel_element.text.strip()
                            
                    except Exception as e:
                        logger.error(f"Error extracting channel info: {str(e)}")
                        channel = ""
                        channel_url = ""
                    
                    # Extract video metadata (views, date)
                    metadata_element = element.find_element(By.CSS_SELECTOR, "#metadata-line")
                    metadata_text = metadata_element.text
                    
                    # Parse views and date
                    views_match = re.search(r"([\d,]+)\s+views", metadata_text)
                    views = views_match.group(1) if views_match else "Unknown"
                    
                    date_match = re.search(r"(?:[\d,]+\s+views\s+)?(.*?)$", metadata_text)
                    date = date_match.group(1) if date_match else "Unknown"
                    
                    # Extract description if available
                    try:
                        description = element.find_element(By.CSS_SELECTOR, "#description-text").text.strip()
                    except:
                        description = ""
                    
                    videos.append({
                        "title": title,
                        "url": video_url,
                        "channel": channel,
                        "channel_url": channel_url,
                        "views": views,
                        "date": date,
                        "description": description
                    })
                    
                except Exception as e:
                    logger.error(f"Error extracting video info: {str(e)}")
                    continue
            
            return {
                "status": "success",
                "results": videos,
                "query": query
            }
            
        finally:
            driver.quit()
            
    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"YouTube search failed: {str(e)}"}