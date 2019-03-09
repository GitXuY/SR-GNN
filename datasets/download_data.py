import pandas as pd
import logging
import sys

logger = logging.getLogger('pandas_gbq')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

start_date = '20181128'
end_date = '20190308'
sites = [83]
locales = ['en_GB']
sample_size = 0.1
minimum_product_views = 1
maximum_product_views = 100

query = f"""
WITH
  first_event_table AS (
  SELECT
    sessionId,
    first_event
  FROM (
    SELECT
      device.sess_cookie AS sessionId,
      MIN(request.start_timestamp) AS first_event,
      SUM(CASE
          WHEN event.type = 'product_visit' THEN 1
          ELSE 0 END) AS product_views
    FROM
      `the-hut-group.site_traffic.elysium*`,
      UNNEST(page.items) AS items
    WHERE (_TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}')
    AND property.site_id in ({str(sites).replace("[", "").replace("]","")})
    AND property.locale in ({str(locales).replace("[", "").replace("]","")})
    GROUP BY sessionId) x
  WHERE
    product_views >= {minimum_product_views}
    AND product_views <= {maximum_product_views}
    AND RAND() < {sample_size}),
  product_views_table AS (
  SELECT
    device.sess_cookie AS sessionId,
    customer.id AS userId,
    COALESCE(items.product_group.id,
      items.product_group.selected_variant.sku) AS itemId,
    request.start_timestamp AS time_stamp
  FROM
    `the-hut-group.site_traffic.elysium*`,
    UNNEST(page.items) AS items
  WHERE
    (_TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}')
    AND property.site_id in ({str(sites).replace("[", "").replace("]","")})
    AND property.locale in ({str(locales).replace("[", "").replace("]","")})
    AND device.is_bot=FALSE
    AND event.type = 'product_visit'
  GROUP BY
    sessionId,
    userId,
    itemId,
    time_stamp)
SELECT
  product_views_table.sessionId,
  product_views_table.userId,
  product_views_table.itemId,
  UNIX_MILLIS(product_views_table.time_stamp) - UNIX_MILLIS(first_event_table.first_event) AS timeframe,
  CAST(product_views_table.time_stamp AS DATE) AS eventdate
FROM
  product_views_table
INNER JOIN
  first_event_table
ON
  product_views_table.sessionId = first_event_table.sessionId
ORDER BY
  sessionId,
  eventdate,
  timeframe
  """

# print(query)

logging.info('Downloading data...')
data = pd.read_gbq(query, dialect='standard')
logging.info('Saving data to disk...')
data.to_csv('thg/train-item-views.csv', index=False, float_format="%.0f")