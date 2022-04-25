
--tables in the database
select * from BusinessCase_Custs limit 10;
select * from BusinessCase_Accts limit 10;
select * from BusinessCase_Tx limit 10;

--review transactions dataset
select
count(*) unique_transactions
,count(distinct customerid) unique_customers
,count(distinct merchantid) unique_merchants
,count(distinct accountid) unique_accounts
,avg(currencyAmount) mean_tx_size
,sum(currencyAmount) total_tx_size
from BusinessCase_Tx

--sql statement to transform transactional data into RFM data
drop table rfm;
CREATE TABLE rfm as
select 
customerid
,start_month
,frequency
,recency
,monetary_value
,T
,frequency*monetary_value as total_sales
from (
	SELECT
	customerid
	,strftime('%m', MIN(Date(originationDateTime))) as start_month
	,COUNT(distinct Date(originationDateTime)) - 1 as frequency
	,julianday(MIN(Date(originationDateTime))) - julianday(MAX(Date(originationDateTime))) as recency
	,round(AVG(currencyAmount), 2) as monetary_value
	,julianday(Date('now')) - julianday(MIN(Date(originationDateTime))) as T
	FROM BusinessCase_Tx
	GROUP BY 1
) a

select * from rfm limit 10;

select * from BusinessCase_Tx limit 10;
select customerid, count(distinct categoryTags) from BusinessCase_Tx group by 1 order by 2 desc

SELECT * from BusinessCase_Custs where id='fe51c153-fbec-4b64-9b00-2530035ef0e1_b1acd4f1-73ea-401b-a8d2-f35648008cae'

select * from BusinessCase_Accts where cust_id='fe51c153-fbec-4b64-9b00-2530035ef0e1_b1acd4f1-73ea-401b-a8d2-f35648008cae'
/*
customer has 2 accounts
DDA - fe51c153-fbec-4b64-9b00-2530035ef0e1_d6fcf650-baa5-4ad5-8a3b-34ac89a1d690
SDA - fe51c153-fbec-4b64-9b00-2530035ef0e1_3b62b558-9ea4-4de9-b5c0-6ec139ee736e
*/

select * from BusinessCase_Tx where 
--customerid='fe51c153-fbec-4b64-9b00-2530035ef0e1_b1acd4f1-73ea-401b-a8d2-f35648008cae'
--and accountId='fe51c153-fbec-4b64-9b00-2530035ef0e1_d6fcf650-baa5-4ad5-8a3b-34ac89a1d690'
--and accountId='fe51c153-fbec-4b64-9b00-2530035ef0e1_3b62b558-9ea4-4de9-b5c0-6ec139ee736e'
--merchantId is NULL
categoryTags is NULL
order by originationDateTime asc
--merchantId can be null for category tags - Taxes, Mortgage and Rent, Income, Transfer
--categoryTags can be NULL
