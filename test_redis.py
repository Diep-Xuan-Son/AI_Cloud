import redis
import numpy as np
import json
import struct

# Create a redis client
redisClient = redis.StrictRedis(host='192.168.6.86',
								port=6400,
								db=0)

# # Add key value pairs to the Redis hash
# redisClient.hset("NumberVsString", "1", "OneOneOneabccba1234")
# redisClient.hset("NumberVsString", "2", "Two")
# redisClient.hset("NumberVsString", "3", "Three")

# # Retrieve the value for a specific key
# print("Value for the key 3 is")
# print(redisClient.hget("NumberVsString", "3"))

# print("The keys present in the Redis hash:");
# print(redisClient.hkeys("NumberVsString"))

# print("The values present in the Redis hash:");
# print(redisClient.hvals("NumberVsString"))
# value_arr = np.asarray(redisClient.hvals("NumberVsString"), dtype=str).tolist()
# value_arr = np.asarray(json.loads(str(value_arr).replace("'","")), dtype=np.float16)
# # value_arr = value_arr.astype(np.float16)
# print(value_arr)

# print("The keys and values present in the Redis hash are:")
# print(redisClient.hgetall("NumberVsString"))

# dt = redisClient.hgetall("NumberVsString")
# print(dt[b"1"])


# print(redisClient.hexists("NumberVsString", "1"))
# redisClient.hdel("NumberVsString", "1")
# redisClient.delete("NumberVsString")

# dt = {
# 		'1': str(np.array([[1,2,3],[4,5,6]], dtype=np.float16).tolist()), 
# 		'2': str(np.array([[10,20,30],[40,50,60]], dtype=np.float16).tolist()), 
# 		'3': str(np.array([[100,200,300],[400,500,600]], dtype=np.float16).tolist())
# 	}
# redisClient.hmset("NumberVsString", dt)

# -----------------------mt2----------------------------
# dt = {
# 		'1': np.array([[1,2,3],[4,5,6]], dtype=np.float16).tobytes(), 
# 		'2': np.array([[10,20,30],[40,50,60]], dtype=np.float16).tobytes(), 
# 		'3': np.array([[100,200,300],[400,500,600]], dtype=np.float16).tobytes()
# 	}
# redisClient.hmset("NumberVsString", dt)
# redisClient.hset("NumberVsString", "4", np.array([[7,8,9],[9,8,7]], dtype=np.float16).tobytes())
# for i in range(1000):
# 	redisClient.hset("NumberVsString", f"{i}", np.array([[i,8,9],[9,8,i]], dtype=np.float16).tobytes())
# value_arr = np.array(redisClient.hvals("NumberVsString"))
# print(np.frombuffer(value_arr, dtype=np.float16).reshape(len(value_arr), 2, 3))
#//////////////////////////////////////////////////////////
