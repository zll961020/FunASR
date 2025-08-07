import pynini
from pynini.lib import utf8, byte
from pynini import cdrewrite

sigma = utf8.VALID_UTF8_CHAR.star

rule1 = pynini.cross("hao4jing1ke1ji4", "浩鲸科技")
rule2 = pynini.cross("hao4jin1ke1ji4", "浩鲸科技")
rule3 = pynini.cross("hou4jin1ke1ji4", "浩鲸科技")



rule = (rule1 | rule2 | rule3).optimize()
rule = cdrewrite(rule, "", "", sigma)

rule.write('hr_resources/replace_iwhalecloud.fst')
