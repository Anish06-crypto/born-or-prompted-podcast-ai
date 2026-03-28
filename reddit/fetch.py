import praw

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# Heuristic quality guards
MIN_SCORE = 100
MIN_TITLE_LENGTH = 25
MIN_COMMENTS = 3
MAX_POSTS_SCANNED = 25

DEFAULT_SUBREDDITS = [
    "technology",
    "science",
    "worldnews",
    "programming",
    "artificial",
]


def _is_quality_post(post) -> bool:
    """Return True if the post clears all heuristic filters."""
    if post.stickied:
        return False
    if post.score < MIN_SCORE:
        return False
    if len(post.title) < MIN_TITLE_LENGTH:
        return False
    if post.num_comments < MIN_COMMENTS:
        return False
    return True


def fetch_episode_seed(subreddit_name: str = None) -> dict | None:
    """
    Fetch the top qualifying post from a subreddit and return a seed dict:
        {topic, position_a_seed, position_b_seed}

    Returns None if credentials are missing, no posts qualify, or any error occurs.
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("[reddit] No credentials configured — skipping Reddit fetch.")
        return None

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

        sub_name = subreddit_name or DEFAULT_SUBREDDITS[0]
        subreddit = reddit.subreddit(sub_name)

        post = None
        for candidate in subreddit.hot(limit=MAX_POSTS_SCANNED):
            if _is_quality_post(candidate):
                post = candidate
                break

        if post is None:
            print(f"[reddit] No qualifying posts found in r/{sub_name}.")
            return None

        # Extract top two substantive comments as opinion seeds
        post.comments.replace_more(limit=0)
        top_comments = [
            c.body.strip()
            for c in post.comments[:10]
            if hasattr(c, "body") and len(c.body.strip()) > 20
        ][:2]

        print(f"[reddit] Topic from r/{sub_name}: {post.title!r}")
        return {
            "topic": post.title,
            "position_a_seed": top_comments[0] if len(top_comments) > 0 else "",
            "position_b_seed": top_comments[1] if len(top_comments) > 1 else "",
        }

    except Exception as e:
        print(f"[reddit] Fetch failed: {e}")
        return None
