from neo4j import GraphDatabase

driver = GraphDatabase.driver("x", auth=("y", "z"))

info = driver.verify_connectivity()
print("Connection info:", info)
print("Host:", driver.default_host)
print("Port:", driver.default_port)
print("Multi-DB support:", driver.supports_multi_db())



records, summary, keys = driver.execute_query(
    "MATCH (n) RETURN count(n) AS node_count",
    database_="neo4j"  # optional if only one DB
)
print("Node count:", records[0]["node_count"])
print("Returned keys:", keys)
