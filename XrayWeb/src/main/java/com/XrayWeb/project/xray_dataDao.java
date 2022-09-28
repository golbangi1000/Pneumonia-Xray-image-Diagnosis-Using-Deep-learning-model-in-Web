package com.XrayWeb.project;
import java.util.List;
import java.util.Map;
import org.mybatis.spring.SqlSessionTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

@Repository 
public class xray_dataDao {
	 @Autowired
	 SqlSessionTemplate sqlSessionTemplate;
	 
	 public int xray_datainsert(Map<String, Object> map) {
		  return this.sqlSessionTemplate.insert("xray_data.insert", map);
		}
	 
	 public List<Map<String, Object>> xray_dataList(Map<String, Object> map) {
		    return this.sqlSessionTemplate.selectList("xray_data.list", map);
		}
	 
}
