"""
Rate Limiting Middleware

Uses Redis to implement a sliding window or fixed window rate limiter.
Protects the API from abuse. Rate limits are applied per-IP for anonymous
users and per-user for authenticated users.
"""

import time
from typing import Any

from fastapi import Request, Response
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from cerebrum.core.cache import get_redis


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        burst: int = 20,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Exclude health check and metrics from rate limiting
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        try:
            redis = get_redis()
        except RuntimeError:
            # If Redis isn't initialized (e.g., during tests without Docker), bypass
            return await call_next(request)

        # Determine client identifier (prefer authenticated user ID, fallback to IP)
        client_id = request.client.host if request.client else "unknown_ip"
        if hasattr(request.state, "user_id") and request.state.user_id:
            client_id = str(request.state.user_id)

        key = f"rate_limit:{client_id}"
        current_minute = int(time.time() / 60)
        window_key = f"{key}:{current_minute}"

        # Atomic increment and expire using pipeline
        async with redis.pipeline() as pipe:
            pipe.incr(window_key)
            pipe.expire(window_key, 120)  # 2 minutes TTL
            result = await pipe.execute()
            
        requests_this_minute = result[0]

        if requests_this_minute > (self.requests_per_minute + self.burst):
            return ORJSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please try again later.",
                },
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute + self.burst - requests_this_minute)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
