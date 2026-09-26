#!/bin/bash

echo "retriever use python:"
which python

api_key=""
cache_file="./search_cache.json"

python search_r1/search/bing_search_server_redis.py --api_key=$api_key \
                                            --cache_file=$cache_file \
                                            --warm_cache_on_start
