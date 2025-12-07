Question 823:  How many conductors are there ? ||| orchestra
SQL:  select count(*) from conductor

Question 824:  Count the number of conductors . ||| orchestra
SQL:  select count(*) from conductor

Question 825:  List the names of conductors in ascending order of age . ||| orchestra
SQL:  select name from conductor order by age asc

Question 826:  What are the names of conductors , ordered by age ? ||| orchestra
SQL:  select name from conductor order by age asc

Question 827:  What are the names of conductors whose nationalities are not `` USA '' ? ||| orchestra
SQL:  select name from conductor where nationality != 'usa'

Question 828:  Return the names of conductors that do not have the nationality `` USA '' . ||| orchestra
SQL:  select name from conductor where nationality != 'usa'

Question 829:  What are the record companies of orchestras in descending order of years in which they were founded ? ||| orchestra
SQL:  select record_company from orchestra order by year_of_founded desc

Question 830:  Return the record companies of orchestras , sorted descending by the years in which they were founded . ||| orchestra
SQL:  select record_company from orchestra order by year_of_founded desc

Question 831:  What is the average attendance of shows ? ||| orchestra
SQL:  select avg(attendance) from show

Question 832:  Return the average attendance across all shows . ||| orchestra
SQL:  select avg(attendance) from show

Question 833:  What are the maximum and minimum share of performances whose type is not `` Live final '' . ||| orchestra
SQL:  select max(share) ,  min(share) from performance where type != "live final"

Question 834:  Return the maximum and minimum shares for performances that do not have the type `` Live final '' . ||| orchestra
SQL:  select max(share) ,  min(share) from performance where type != "live final"

Question 835:  How many different nationalities do conductors have ? ||| orchestra
SQL:  select count(distinct nationality) from conductor

Question 836:  Count the number of different nationalities of conductors . ||| orchestra
SQL:  select count(distinct nationality) from conductor

Question 837:  List names of conductors in descending order of years of work . ||| orchestra
SQL:  select name from conductor order by year_of_work desc

Question 838:  What are the names of conductors , sorted descending by the number of years they have worked ? ||| orchestra
SQL:  select name from conductor order by year_of_work desc

Question 839:  List the name of the conductor with the most years of work . ||| orchestra
SQL:  select name from conductor order by year_of_work desc limit 1

Question 840:  What is the name of the conductor who has worked the greatest number of years ? ||| orchestra
SQL:  select name from conductor order by year_of_work desc limit 1

Question 841:  Show the names of conductors and the orchestras they have conducted . ||| orchestra
SQL:  select t1.name ,  t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id

Question 842:  What are the names of conductors as well as the corresonding orchestras that they have conducted ? ||| orchestra
SQL:  select t1.name ,  t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id

Question 843:  Show the names of conductors that have conducted more than one orchestras . ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id having count(*)  >  1

Question 844:  What are the names of conductors who have conducted at more than one orchestra ? ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id having count(*)  >  1

Question 845:  Show the name of the conductor that has conducted the most number of orchestras . ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id order by count(*) desc limit 1

Question 846:  What is the name of the conductor who has conducted the most orchestras ? ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id order by count(*) desc limit 1

Question 847:  Please show the name of the conductor that has conducted orchestras founded after 2008 . ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id where year_of_founded  >  2008

Question 848:  What are the names of conductors who have conducted orchestras founded after the year 2008 ? ||| orchestra
SQL:  select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id where year_of_founded  >  2008

Question 849:  Please show the different record companies and the corresponding number of orchestras . ||| orchestra
SQL:  select record_company ,  count(*) from orchestra group by record_company

Question 850:  How many orchestras does each record company manage ? ||| orchestra
SQL:  select record_company ,  count(*) from orchestra group by record_company

Question 851:  Please show the record formats of orchestras in ascending order of count . ||| orchestra
SQL:  select major_record_format from orchestra group by major_record_format order by count(*) asc

Question 852:  What are the major record formats of orchestras , sorted by their frequency ? ||| orchestra
SQL:  select major_record_format from orchestra group by major_record_format order by count(*) asc

Question 853:  List the record company shared by the most number of orchestras . ||| orchestra
SQL:  select record_company from orchestra group by record_company order by count(*) desc limit 1

Question 854:  What is the record company used by the greatest number of orchestras ? ||| orchestra
SQL:  select record_company from orchestra group by record_company order by count(*) desc limit 1

Question 855:  List the names of orchestras that have no performance . ||| orchestra
SQL:  select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)

Question 856:  What are the orchestras that do not have any performances ? ||| orchestra
SQL:  select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)

Question 857:  Show the record companies shared by orchestras founded before 2003 and after 2003 . ||| orchestra
SQL:  select record_company from orchestra where year_of_founded  <  2003 intersect select record_company from orchestra where year_of_founded  >  2003

Question 858:  What are the record companies that are used by both orchestras founded before 2003 and those founded after 2003 ? ||| orchestra
SQL:  select record_company from orchestra where year_of_founded  <  2003 intersect select record_company from orchestra where year_of_founded  >  2003

Question 859:  Find the number of orchestras whose record format is `` CD '' or `` DVD '' . ||| orchestra
SQL:  select count(*) from orchestra where major_record_format  =  "cd" or major_record_format  =  "dvd"

Question 860:  Count the number of orchestras that have CD or DVD as their record format . ||| orchestra
SQL:  select count(*) from orchestra where major_record_format  =  "cd" or major_record_format  =  "dvd"

Question 861:  Show the years in which orchestras that have given more than one performance are founded . ||| orchestra
SQL:  select year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id  =  t2.orchestra_id group by t2.orchestra_id having count(*)  >  1

Question 862:  What are years of founding for orchestras that have had more than a single performance ? ||| orchestra
SQL:  select year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id  =  t2.orchestra_id group by t2.orchestra_id having count(*)  >  1

