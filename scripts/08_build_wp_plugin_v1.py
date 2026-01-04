#!/usr/bin/env python3
"""
08_build_wp_plugin_v1.py - Generate WordPress Plugin for YouTube Semantic Search
=================================================================================

OVERVIEW
--------
This script generates a complete WordPress plugin for embedding YouTube
semantic search on a WordPress site. The plugin connects to your Flask API
server (created by 07_local_POC_v1.py) to perform searches.

Run this AFTER:
  1. Completing the data pipeline (scripts 01-06)
  2. Generating the Flask server (script 07)
  3. Testing that the server works locally or is deployed

The script creates a complete WordPress plugin in:
  wp-plugin/{project-slug}-ai-search/

WHAT GETS GENERATED
-------------------
The script creates the following files:

  {project-slug}-ai-search/
  ├── {project-slug}-ai-search.php    # Main plugin file
  │   - Plugin header (name, version, author)
  │   - Activation/deactivation hooks
  │   - Script/style enqueueing
  │   - AJAX action handlers
  │   - Admin menu registration
  │
  ├── includes/
  │   ├── class-admin.php             # Admin settings page
  │   │   - Settings page UI
  │   │   - API URL configuration
  │   │   - API Key configuration
  │   │   - Results per page setting
  │   │   - AI summary toggle
  │   │   - Connection test button
  │   │
  │   ├── class-shortcode.php         # Search shortcode
  │   │   - [channel_search] shortcode
  │   │   - Renders search interface
  │   │   - Handles AJAX search requests
  │   │   - Displays video thumbnails
  │   │   - Shows AI summaries
  │   │
  │   └── class-analytics.php         # Click tracking
  │       - Records result clicks
  │       - Tracks search queries
  │       - Batches analytics via sendBeacon
  │       - Stores in WordPress database
  │
  ├── public/
  │   ├── css/
  │   │   └── search.css              # Frontend styles
  │   │       - Search box styling
  │   │       - Video thumbnail grid
  │   │       - Result cards
  │   │       - AI summary box
  │   │       - Loading spinner
  │   │       - Responsive design
  │   │
  │   └── js/
  │       └── search.js               # Frontend JavaScript
  │           - Search form handling
  │           - AJAX requests to WP backend
  │           - Result rendering
  │           - Video thumbnail display
  │           - AI summary loading
  │           - Click tracking
  │           - Load more pagination
  │
  ├── admin/
  │   └── js/
  │       └── admin.js                # Admin JavaScript
  │           - Connection test
  │           - Settings validation
  │
  └── README.md                       # Plugin documentation

PLUGIN FEATURES
---------------
1. SEARCH SHORTCODE
   Embed search anywhere with: [channel_search]
   
   Shortcode attributes:
   - results_per_page="10"   (default from settings)
   - show_summary="true"     (default from settings)
   - placeholder="Search..." (search box placeholder text)

2. ADMIN SETTINGS
   Settings → {Channel Name} Search
   - API URL: Your server endpoint (e.g., https://search.yoursite.com)
   - API Key: Authentication key for API requests
   - Results per page: 5, 10, 15, or 20
   - Enable AI Summary: Toggle AI-generated summaries
   - Test Connection: Verify API is reachable

3. VIDEO DISPLAY
   - Thumbnail grid for unique videos in results
   - Timestamp badges showing where match occurs
   - Click-to-play links to YouTube with timestamp

4. AI SUMMARIES
   - Optional AI-generated summary of search results
   - Shows source videos
   - Styled summary box

5. CLICK ANALYTICS
   - Tracks which results users click
   - Batched reporting to minimize requests
   - Viewable in admin dashboard

PREREQUISITES
-------------
1. Flask server running (from 07_local_POC_v1.py)
2. API key configured in server's .env
3. WordPress site with admin access

USAGE
-----
    cd scripts
    python 08_build_wp_plugin_v1.py
    
    # Plugin files will be generated in:
    # ../wp-plugin/{project-slug}-ai-search/
    
    # To install:
    # 1. Zip the plugin folder
    # 2. WordPress Admin → Plugins → Add New → Upload Plugin
    # 3. Activate the plugin
    # 4. Go to Settings → {Channel Name} Search
    # 5. Enter your API URL and API Key

CONFIG IMPORTS USED
-------------------
From config.py:
  - PROJECT_NAME, PROJECT_SLUG
  - CHANNEL_HANDLE, CHANNEL_DISPLAY_NAME
  - WP_PLUGIN_DIR, WP_PLUGIN_INCLUDES_DIR
  - WP_PLUGIN_PUBLIC_DIR, WP_PLUGIN_ADMIN_DIR
  - SERVER_PORT (for default localhost URL)

API ENDPOINTS CALLED BY PLUGIN
------------------------------
The generated plugin makes these API calls to your Flask server:

  POST /api/v1/search
  Headers: X-API-Key: {configured_key}
  Body: {"query": "search text", "top_k": 10}
  
  POST /api/v1/summarize
  Headers: X-API-Key: {configured_key}
  Body: {"query": "search text", "results": [...]}
  
  GET /api/v1/health
  Headers: X-API-Key: {configured_key}
  (Used for connection test in admin)

WORDPRESS REQUIREMENTS
----------------------
- WordPress 5.0 or higher
- PHP 7.4 or higher
- AJAX/REST capabilities (standard in WP)

SECURITY CONSIDERATIONS
-----------------------
1. API Key is stored in WordPress options (wp_options table)
2. All AJAX requests use WordPress nonces for CSRF protection
3. API requests are server-side (key not exposed to browser)
4. Input sanitization on all user inputs

CUSTOMIZATION AFTER GENERATION
------------------------------
The generated plugin uses neutral styling. To customize:

1. Colors/Branding: Edit public/css/search.css
2. Result Layout: Edit class-shortcode.php render method
3. API Behavior: Edit class-shortcode.php handle_search method
4. Analytics: Extend class-analytics.php

Generated: Part of YouTube AI Semantic Search pipeline
"""

import sys
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from config import (
        PROJECT_NAME, PROJECT_SLUG,
        CHANNEL_HANDLE, CHANNEL_DISPLAY_NAME,
        WP_PLUGIN_DIR, WP_PLUGIN_INCLUDES_DIR,
        WP_PLUGIN_PUBLIC_DIR, WP_PLUGIN_ADMIN_DIR,
        SERVER_PORT,
        ensure_directories,
    )
except ImportError as e:
    print(f"ERROR: Could not import from config.py: {e}")
    print("Make sure you're running this from the scripts/ directory")
    print("and that config.py exists in the project root.")
    sys.exit(1)


# =============================================================================
# PLUGIN FILE TEMPLATES
# =============================================================================
# These templates will be populated with project-specific values when
# the full implementation is built. For now, they serve as documentation
# of what will be generated.

MAIN_PLUGIN_PHP_TEMPLATE = '''<?php
/**
 * Plugin Name: {channel_display_name} AI Search
 * Plugin URI: https://github.com/yourusername/{project_slug}-ai-search
 * Description: Semantic search across {channel_display_name}'s YouTube transcripts
 * Version: 1.0.0
 * Author: Your Name
 * License: GPL v2 or later
 * Text Domain: {project_slug}-ai-search
 */

// Prevent direct access
if (!defined('ABSPATH')) {{
    exit;
}}

// Plugin constants
define('{PLUGIN_PREFIX}_VERSION', '1.0.0');
define('{PLUGIN_PREFIX}_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('{PLUGIN_PREFIX}_PLUGIN_URL', plugin_dir_url(__FILE__));

// Include required files
require_once {PLUGIN_PREFIX}_PLUGIN_DIR . 'includes/class-admin.php';
require_once {PLUGIN_PREFIX}_PLUGIN_DIR . 'includes/class-shortcode.php';
require_once {PLUGIN_PREFIX}_PLUGIN_DIR . 'includes/class-analytics.php';

// Initialize plugin
// ... (hooks, filters, etc.)
'''

ADMIN_CLASS_PHP_TEMPLATE = '''<?php
/**
 * Admin settings page for {channel_display_name} AI Search
 * 
 * Handles:
 * - Settings page registration
 * - API URL and Key configuration
 * - Connection testing
 * - Results per page setting
 * - AI summary toggle
 */

class {PluginPrefix}_Admin {{
    // Settings page implementation
    // ... 
}}
'''

SHORTCODE_CLASS_PHP_TEMPLATE = '''<?php
/**
 * Search shortcode for {channel_display_name} AI Search
 * 
 * Usage: [{project_slug}_search]
 * 
 * Attributes:
 * - results_per_page: Number of results (default: from settings)
 * - show_summary: Show AI summary (default: from settings)
 * - placeholder: Search input placeholder text
 */

class {PluginPrefix}_Shortcode {{
    // Shortcode implementation
    // ...
}}
'''

ANALYTICS_CLASS_PHP_TEMPLATE = '''<?php
/**
 * Analytics tracking for {channel_display_name} AI Search
 * 
 * Tracks:
 * - Search queries
 * - Result clicks
 * - Video views
 * 
 * Data stored in WordPress database for admin viewing.
 */

class {PluginPrefix}_Analytics {{
    // Analytics implementation
    // ...
}}
'''

SEARCH_CSS_TEMPLATE = '''/**
 * {channel_display_name} AI Search - Frontend Styles
 * 
 * Sections:
 * 1. Search Box
 * 2. Loading States
 * 3. AI Summary
 * 4. Video Grid
 * 5. Result Cards
 * 6. Responsive
 */

/* Search container */
.{project_slug}-search-container {{
    max-width: 900px;
    margin: 0 auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

/* ... more styles ... */
'''

SEARCH_JS_TEMPLATE = '''/**
 * {channel_display_name} AI Search - Frontend JavaScript
 * 
 * Handles:
 * - Search form submission
 * - AJAX requests via WordPress
 * - Result rendering
 * - Video thumbnail display
 * - AI summary loading
 * - Click tracking
 * - Load more pagination
 */

(function($) {{
    'use strict';
    
    // Plugin initialization
    // ...
    
}})(jQuery);
'''


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Generate WordPress plugin files."""
    print()
    print("=" * 60)
    print("08_build_wp_plugin.py - Generate WordPress Plugin")
    print("=" * 60)
    print()
    print(f"Project:     {PROJECT_NAME}")
    print(f"Plugin Slug: {PROJECT_SLUG}-ai-search")
    print(f"Channel:     {CHANNEL_DISPLAY_NAME} ({CHANNEL_HANDLE})")
    print(f"Output Dir:  {WP_PLUGIN_DIR}")
    print()
    
    # Ensure directories exist
    ensure_directories()
    WP_PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    WP_PLUGIN_INCLUDES_DIR.mkdir(parents=True, exist_ok=True)
    (WP_PLUGIN_PUBLIC_DIR / "css").mkdir(parents=True, exist_ok=True)
    (WP_PLUGIN_PUBLIC_DIR / "js").mkdir(parents=True, exist_ok=True)
    (WP_PLUGIN_ADMIN_DIR / "js").mkdir(parents=True, exist_ok=True)
    
    # Generate plugin prefix (uppercase, underscores)
    plugin_prefix = PROJECT_SLUG.upper().replace('-', '_')
    plugin_class_prefix = ''.join(word.capitalize() for word in PROJECT_SLUG.split('-'))
    
    print("This script will generate:")
    print(f"  - {PROJECT_SLUG}-ai-search.php (main plugin file)")
    print(f"  - includes/class-admin.php")
    print(f"  - includes/class-shortcode.php")
    print(f"  - includes/class-analytics.php")
    print(f"  - public/css/search.css")
    print(f"  - public/js/search.js")
    print(f"  - admin/js/admin.js")
    print()
    
    # TODO: Implement full file generation
    # For now, create placeholder files with comments
    
    print("-" * 60)
    print("NOTE: Full plugin generation not yet implemented.")
    print("This is a placeholder showing what will be created.")
    print("-" * 60)
    print()
    print("The plugin structure has been created at:")
    print(f"  {WP_PLUGIN_DIR}")
    print()
    print("To complete the plugin, the following files need content:")
    print(f"  - {PROJECT_SLUG}-ai-search.php")
    print(f"  - includes/class-admin.php")
    print(f"  - includes/class-shortcode.php")
    print(f"  - includes/class-analytics.php")
    print(f"  - public/css/search.css")
    print(f"  - public/js/search.js")
    print()
    print("See the README.md in the plugin directory for details.")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
