select  distinct(i_product_name)
 from item i1
 where i_manufact_id between 814 and 814+40 
   and (select count(*) as item_cnt
        from item
        where (i_manufact = i1.i_manufact and
        ((i_category = 'Women' and 
        (i_color = 'midnight' or i_color = 'light') and 
        (i_units = 'Gram' or i_units = 'Case') and
        (i_size = 'medium' or i_size = 'extra large')
        ) or
        (i_category = 'Women' and
        (i_color = 'lavender' or i_color = 'green') and
        (i_units = 'Lb' or i_units = 'Pallet') and
        (i_size = 'petite' or i_size = 'N/A')
        ) or
        (i_category = 'Men' and
        (i_color = 'pale' or i_color = 'burlywood') and
        (i_units = 'Box' or i_units = 'Oz') and
        (i_size = 'large' or i_size = 'economy')
        ) or
        (i_category = 'Men' and
        (i_color = 'antique' or i_color = 'pink') and
        (i_units = 'Bundle' or i_units = 'Unknown') and
        (i_size = 'medium' or i_size = 'extra large')
        ))) or
       (i_manufact = i1.i_manufact and
        ((i_category = 'Women' and 
        (i_color = 'drab' or i_color = 'deep') and 
        (i_units = 'Gross' or i_units = 'Dram') and
        (i_size = 'medium' or i_size = 'extra large')
        ) or
        (i_category = 'Women' and
        (i_color = 'lace' or i_color = 'steel') and
        (i_units = 'Bunch' or i_units = 'Cup') and
        (i_size = 'petite' or i_size = 'N/A')
        ) or
        (i_category = 'Men' and
        (i_color = 'cornsilk' or i_color = 'saddle') and
        (i_units = 'Tbl' or i_units = 'N/A') and
        (i_size = 'large' or i_size = 'economy')
        ) or
        (i_category = 'Men' and
        (i_color = 'chiffon' or i_color = 'yellow') and
        (i_units = 'Pound' or i_units = 'Ounce') and
        (i_size = 'medium' or i_size = 'extra large')
        )))) > 0
 order by i_product_name
 limit 100;
