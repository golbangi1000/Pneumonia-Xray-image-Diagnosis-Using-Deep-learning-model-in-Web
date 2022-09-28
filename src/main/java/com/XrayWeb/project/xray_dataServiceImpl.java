package com.XrayWeb.project;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class xray_dataServiceImpl implements xray_dataService {
	@Autowired
	xray_dataDao xray_dataDao; 
	
	@Override
	public String xray_datacreate(Map<String, Object> map) {
		int affectRowCount = this.xray_dataDao.xray_datainsert(map); 
	    
		if (affectRowCount ==  1) {
	        return map.get("member_Id").toString(); 
	    }
	    return null; 
	}

	@Override
	public List<Map<String, Object>> xray_dataList(Map<String, Object> map) {
		// TODO Auto-generated method stub
		return this.xray_dataDao.xray_dataList(map);
	}

}
