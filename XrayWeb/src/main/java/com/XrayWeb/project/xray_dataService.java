package com.XrayWeb.project;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Service;

public interface xray_dataService {
	String xray_datacreate(Map<String, Object> map);
	List<Map<String, Object>> xray_dataList(Map<String, Object> map);

}
