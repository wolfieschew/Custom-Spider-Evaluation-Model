Question 1001:  How many singers are there ? ||| singer
SQL:  select count(*) from singer

Question 1002:  What is the count of singers ? ||| singer
SQL:  select count(*) from singer

Question 1003:  List the name of singers in ascending order of net worth . ||| singer
SQL:  select name from singer order by net_worth_millions asc

Question 1004:  What are the names of singers ordered by ascending net worth ? ||| singer
SQL:  select name from singer order by net_worth_millions asc

Question 1005:  What are the birth year and citizenship of singers ? ||| singer
SQL:  select birth_year ,  citizenship from singer

Question 1006:  What are the birth years and citizenships of the singers ? ||| singer
SQL:  select birth_year ,  citizenship from singer

Question 1007:  List the name of singers whose citizenship is not `` France '' . ||| singer
SQL:  select name from singer where citizenship != "france"

Question 1008:  What are the names of the singers who are not French citizens ? ||| singer
SQL:  select name from singer where citizenship != "france"

Question 1009:  Show the name of singers whose birth year is either 1948 or 1949 ? ||| singer
SQL:  select name from singer where birth_year  =  1948 or birth_year  =  1949

Question 1010:  What are the names of the singers whose birth years are either 1948 or 1949 ? ||| singer
SQL:  select name from singer where birth_year  =  1948 or birth_year  =  1949

Question 1011:  What is the name of the singer with the largest net worth ? ||| singer
SQL:  select name from singer order by net_worth_millions desc limit 1

Question 1012:  What is the name of the singer who is worth the most ? ||| singer
SQL:  select name from singer order by net_worth_millions desc limit 1

Question 1013:  Show different citizenship of singers and the number of singers of each citizenship . ||| singer
SQL:  select citizenship ,  count(*) from singer group by citizenship

Question 1014:  For each citizenship , how many singers are from that country ? ||| singer
SQL:  select citizenship ,  count(*) from singer group by citizenship

Question 1015:  Please show the most common citizenship of singers . ||| singer
SQL:  select citizenship from singer group by citizenship order by count(*) desc limit 1

Question 1016:  What is the most common singer citizenship ? ||| singer
SQL:  select citizenship from singer group by citizenship order by count(*) desc limit 1

Question 1017:  Show different citizenships and the maximum net worth of singers of each citizenship . ||| singer
SQL:  select citizenship ,  max(net_worth_millions) from singer group by citizenship

Question 1018:  For each citizenship , what is the maximum net worth ? ||| singer
SQL:  select citizenship ,  max(net_worth_millions) from singer group by citizenship

Question 1019:  Show titles of songs and names of singers . ||| singer
SQL:  select t2.title ,  t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id

Question 1020:  What are the song titles and singer names ? ||| singer
SQL:  select t2.title ,  t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id

Question 1021:  Show distinct names of singers that have songs with sales more than 300000 . ||| singer
SQL:  select distinct t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id where t2.sales  >  300000

Question 1022:  what are the different names of the singers that have sales more than 300000 ? ||| singer
SQL:  select distinct t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id where t2.sales  >  300000

Question 1023:  Show the names of singers that have more than one song . ||| singer
SQL:  select t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name having count(*)  >  1

Question 1024:  What are the names of the singers that have more than one songs ? ||| singer
SQL:  select t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name having count(*)  >  1

Question 1025:  Show the names of singers and the total sales of their songs . ||| singer
SQL:  select t1.name ,  sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name

Question 1026:  For each singer name , what is the total sales for their songs ? ||| singer
SQL:  select t1.name ,  sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name

Question 1027:  List the name of singers that do not have any song . ||| singer
SQL:  select name from singer where singer_id not in (select singer_id from song)

Question 1028:  What is the sname of every sing that does not have any song ? ||| singer
SQL:  select name from singer where singer_id not in (select singer_id from song)

Question 1029:  Show the citizenship shared by singers with birth year before 1945 and after 1955 . ||| singer
SQL:  select citizenship from singer where birth_year  <  1945 intersect select citizenship from singer where birth_year  >  1955

Question 1030:  What are the citizenships that are shared by singers with a birth year before 1945 and after 1955 ? ||| singer
SQL:  select citizenship from singer where birth_year  <  1945 intersect select citizenship from singer where birth_year  >  1955

