import asyncio
import os
import getpass
from colorama import Fore, Style
from cookbook.swarm import Swarm, SwarmConfig, SwarmMode, SwarmPlan, AgentSpec, ToolRegistry
from cookbook.swarm.instagrapi_bridge import InstagrapiBridge, get_instagrapi_tools
import sys

def prompt_credentials():
    print(Fore.CYAN + "=== Instagrapi Swarm Setup ===" + Style.RESET_ALL)
    username = os.environ.get("IG_USERNAME")
    password = os.environ.get("IG_PASSWORD")
    if not username:
        username = input("Instagram Username: ")
    if not password:
        password = getpass.getpass("Instagram Password: ")
    return username, password

async def main():
    username, password = prompt_credentials()
    if not username or not password:
        print("Missing credentials. Exiting.")
        sys.exit(1)
        
    print(Fore.YELLOW + "Initializing instagrapi client (may take a few seconds)..." + Style.RESET_ALL)
    try:
        bridge = InstagrapiBridge(username=username, password=password)
    except Exception as e:
        print(f"Failed to setup bridge: {e}")
        sys.exit(1)
        
    tools = get_instagrapi_tools(bridge)
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    
    cfg = SwarmConfig(
        max_agents=5,
        default_model="deepseek-chat",
        api_base="http://localhost:8000/v1", # DeepSeek proxy
        api_key="not-needed",
        agent_timeout=800.0,
        tool_timeout=120.0
    )
    
    # Let's craft the perfect anti-hallucination agent prompt to match original request
    
    scout = AgentSpec(
        role="Instagram Lead Scout",
        task=(
            "Use ig_hashtag_feed to find recent posts for hashtags like '#lagosbusiness' and '#nigerianstartup'.\n"
            "For promising businesses, use ig_get_profile to check their bio.\n"
            "If they seem to need web design/development, save them using ig_save_lead.\n"
            "CRITICAL: \n"
            " - DO NOT make up fake user data.\n"
            " - DO NOT hallucinate IDs like '123456789'. You MUST extract real numeric IDs from tool responses.\n"
            " - DO NOT simulate tool calls. Wait for real tool execution.\n"
        ),
        tools=["ig_hashtag_feed", "ig_get_profile", "ig_save_lead"],
        priority=2,
    )
    
    qualifier = AgentSpec(
        role="Lead Qualifier",
        task=(
            "Review leads sent from scouts or use ig_search_users yourself if fewer than 5 exist.\n"
            "Save the best ones using ig_save_lead with a nice draft DM in the notes.\n"
            "CRITICAL:\n"
            " - Wait for real tool data.\n"
            " - DO NOT make up placeholders or usernames.\n"
            " - REAL Instagram data only.\n"
        ),
        tools=["ig_search_users", "ig_get_profile", "ig_save_lead", "ig_campaign_history"],
        depends_on=[scout.agent_id],
        priority=1,
    )

    plan = SwarmPlan(
        goal="Find 5 highly qualified leads for web design services in Lagos using real Instagram API data.",
        agents=[scout, qualifier],
        strategy="Scout via hashtags -> Qualify and verify via direct search -> Save Real Leads."
    )

    print(Fore.GREEN + f"\nStarting Instagrapi Swarm Execution. Check IG ledger for saves!" + Style.RESET_ALL)
    
    result = await Swarm(
        plan.goal,
        config=cfg,
        mode=SwarmMode.MANUAL,
        plan=plan,
        tool_registry=registry,
        verbose=True
    )
    
    print("\nResults:")
    print(result.synthesis)

if __name__ == "__main__":
    asyncio.run(main())
